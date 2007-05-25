#include <memory>
#include <string>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "FastSimulation/Tracking/interface/GSTrackCandidateMaker.h"
#include "RecoTracker/CkfPattern/interface/TransientInitialStateEstimator.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"


//for debug only 
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h" 
#include "DataFormats/SiStripDetId/interface/TIBDetId.h" 
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h" 
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "FastSimulation/ParticlePropagator/interface/ParticlePropagator.h"
//

//#define FAMOS_DEBUG

GSTrackCandidateMaker::GSTrackCandidateMaker(edm::ParameterSet const& conf) : 
  conf_(conf)
{  
#ifdef FAMOS_DEBUG
    std::cout << "GSTrackCandidateMaker created" << std::endl;
#endif
    produces<TrackCandidateCollection>();

    

}

  
// Virtual destructor needed.
GSTrackCandidateMaker::~GSTrackCandidateMaker() {

  // do nothing
#ifdef FAMOS_DEBUG
  std::cout << "GSTrackCandidateMaker destructed" << std::endl;
#endif

} 
 
void 
GSTrackCandidateMaker::beginJob (edm::EventSetup const & es) {

  //services
  //  es.get<TrackerRecoGeometryRecord>().get(theGeomSearchTracker);

  edm::ESHandle<MagneticField>          magField;
  edm::ESHandle<TrackerGeometry>        geometry;

  es.get<IdealMagneticFieldRecord>().get(magField);
  es.get<TrackerDigiGeometryRecord>().get(geometry);

  theMagField = &(*magField);
  theGeometry = &(*geometry);

  // The smallest true pT for a track candidate
  pTMin = conf_.getParameter<double>("pTMin");
  pTMin *= pTMin;  // Cut is done of perp2() - CPU saver

  // The smallest number of Rec Hits for a track candidate
  minRecHits = conf_.getParameter<unsigned int>("MinRecHits");

  // The smallest true impact parameters (d0 and z0) for a track candidate
  maxD0 = conf_.getParameter<double>("MaxD0");
  maxZ0 = conf_.getParameter<double>("MaxZ0");

}
  
  // Functions that gets called by framework every event
void 
GSTrackCandidateMaker::produce(edm::Event& e, const edm::EventSetup& es) {        

#ifdef FAMOS_DEBUG
  std::cout << "################################################################" << std::endl;
  std::cout << " GSTrackCandidateMaker produce init " << std::endl;
#endif

  unsigned nSimTracks = 0;
  unsigned nTracksWithHits = 0;
  unsigned nTracksWithPT = 0;
  unsigned nTracksWithD0Z0 = 0;
  unsigned nTrackCandidates = 0;
  PTrajectoryStateOnDet initialState;
  
  std::auto_ptr<TrackCandidateCollection> output(new TrackCandidateCollection);    
  
  edm::Handle<edm::SimTrackContainer> theSimTracks;
  e.getByType<edm::SimTrackContainer>(theSimTracks);
  
  edm::Handle<edm::SimVertexContainer> theSimVtx;
  e.getByType(theSimVtx);

#ifdef FAMOS_DEBUG
  std::cout << " Step A: SimTracks found " << theSimTracks->size() << std::endl;
#endif
  
  edm::Handle<SiTrackerGSRecHit2DCollection> theGSRecHits;
  std::string hitProducer = conf_.getParameter<std::string>("HitProducer");
  e.getByLabel(hitProducer, theGSRecHits);
  
  // No tracking attempted if no hits !
#ifdef FAMOS_DEBUG
  std::cout << " Step B: Full GS RecHits found " << theGSRecHits->size() << std::endl;
#endif
  if(theGSRecHits->size() == 0) return;
  
  
#ifdef FAMOS_DEBUG
  std::cout << " Step C: Loop over the RecHits, track by track " << std::endl;
#endif

  // The vector of simTrack Id's carrying GSRecHits
  const std::vector<unsigned> theSimTrackIds = theGSRecHits->ids();

  // loop over SimTrack Id's
  for ( unsigned tkId=0;  tkId != theSimTrackIds.size(); ++tkId ) {

    ++nSimTracks;
    unsigned simTrackId = theSimTrackIds[tkId];
    const SimTrack& theSimTrack = (*theSimTracks)[simTrackId]; 

    SiTrackerGSRecHit2DCollection::range theRecHitRange = theGSRecHits->get(simTrackId);
    SiTrackerGSRecHit2DCollection::const_iterator theRecHitRangeIteratorBegin = theRecHitRange.first;
    SiTrackerGSRecHit2DCollection::const_iterator theRecHitRangeIteratorEnd   = theRecHitRange.second;
    SiTrackerGSRecHit2DCollection::const_iterator iterRecHit;
    SiTrackerGSRecHit2DCollection::const_iterator iterRecHit2;

    // Request a minimum number of RecHits
    unsigned numberOfRecHits = 0;
    for ( iterRecHit = theRecHitRangeIteratorBegin; 
	  iterRecHit != theRecHitRangeIteratorEnd; 
	  ++iterRecHit) ++numberOfRecHits;
    if ( numberOfRecHits < minRecHits ) continue;
    ++nTracksWithHits;

    // Request a minimum pT for the sim track
    if ( theSimTrack.momentum().perp2() < pTMin ) continue;
    ++nTracksWithPT;

    // Check that the sim track comes from the main vertex (loose cut)
    int vertexIndex = theSimTrack.vertIndex();
    const SimVertex& theSimVertex = (*theSimVtx)[vertexIndex]; 
    
    BaseParticlePropagator theParticle = 
      BaseParticlePropagator( 
        RawParticle(XYZTLorentzVector(theSimTrack.momentum().px(),
				      theSimTrack.momentum().py(),
				      theSimTrack.momentum().pz(),
				      theSimTrack.momentum().e()),
		    XYZTLorentzVector(theSimVertex.position().x(),
				      theSimVertex.position().y(),
				      theSimVertex.position().z(),
				      theSimVertex.position().t())),
	0.,0.,4.);
    theParticle.setCharge((*theSimTracks)[simTrackId].charge());
    if ( theParticle.xyImpactParameter() > maxD0 ) continue;
    if ( fabs( theParticle.zImpactParameter() ) > maxZ0 ) continue;
    ++nTracksWithD0Z0;


    //Check Seeding requirements (GlobalPixel only for now)
    
    bool compatible = false;
    const SiTrackerGSRecHit2D *hit1;
    const SiTrackerGSRecHit2D *hit2;
    for ( iterRecHit = theRecHitRangeIteratorBegin; 
	  iterRecHit != theRecHitRangeIteratorEnd; 
	  ++iterRecHit) {
      hit1 = &(*iterRecHit);
      if((unsigned int)(hit1->geographicalId().subdetId())== PixelSubdetector::PixelBarrel || 
	 (unsigned int)(hit1->geographicalId().subdetId())== PixelSubdetector::PixelEndcap){
	const DetId& detId = hit1->geographicalId();
	const GeomDet* geomDet( theGeometry->idToDet(detId) );
	gpos1 = geomDet->surface().toGlobal(hit1->localPosition());
	for ( iterRecHit2 = iterRecHit+1; iterRecHit2 != theRecHitRangeIteratorEnd; ++iterRecHit2) {
	  hit2 = &(*iterRecHit2);
	  if((unsigned int)hit2->geographicalId().subdetId()== PixelSubdetector::PixelBarrel || 
	     (unsigned int)hit2->geographicalId().subdetId()== PixelSubdetector::PixelEndcap){
	    const DetId& detId = hit2->geographicalId();
	    const GeomDet* geomDet( theGeometry->idToDet(detId) );
	    gpos2 = geomDet->surface().toGlobal(hit2->localPosition());

	    compatible = compatibleWithVertex(gpos1,gpos2);
	    
	    if(compatible) break;
	  }
	}
	if(compatible) break;
      }
    }
    if(!compatible) continue;
    
    // Create OwnVector with sorted GSRecHit's
    edm::OwnVector<TrackingRecHit> recHits;
    for ( iterRecHit = theRecHitRangeIteratorBegin; 
	   iterRecHit != theRecHitRangeIteratorEnd; 
	   ++iterRecHit) {
      const DetId& detId =  iterRecHit->geographicalId();
      const GeomDet* geomDet( theGeometry->idToDet(detId) );
      TrackingRecHit* aTrackingRecHit = 
      	GenericTransientTrackingRecHit::build(geomDet,&(*iterRecHit))->hit()->clone();
      recHits.push_back(aTrackingRecHit);
      
#ifdef FAMOS_DEBUG
      unsigned int subdetId = detId.subdetId(); 
      int layerNumber=0;
      int ringNumber = 0;
      int stereo = 0;
      if ( subdetId == StripSubdetector::TIB) { 
	  TIBDetId tibid(detId.rawId()); 
	  layerNumber = tibid.layer();
	  stereo = tibid.stereo();
      } else if ( subdetId ==  StripSubdetector::TOB ) { 
	  TOBDetId tobid(detId.rawId()); 
	  layerNumber = tobid.layer();
	  stereo = tobid.stereo();
      } else if ( subdetId ==  StripSubdetector::TID) { 
	  TIDDetId tidid(detId.rawId());
	  layerNumber = tidid.wheel();
	  ringNumber = tidid.ring();
	  stereo = tidid.stereo();
      } else if ( subdetId ==  StripSubdetector::TEC ) { 
	  TECDetId tecid(detId.rawId()); 
	  layerNumber = tecid.wheel(); 
	  ringNumber = tecid.ring();
	  stereo = tecid.stereo();
      } else if ( subdetId ==  PixelSubdetector::PixelBarrel ) { 
	  PXBDetId pxbid(detId.rawId()); 
	  layerNumber = pxbid.layer();  
	  stereo = 1;
      } else if ( subdetId ==  PixelSubdetector::PixelEndcap ) { 
	  PXFDetId pxfid(detId.rawId()); 
	  layerNumber = pxfid.disk();  
	  stereo = 1;
      }

      std::cout << "Added RecHit from detid " << detId.rawId() 
		<< " subdet = " << subdetId 
		<< " layer = " << layerNumber 
		<< " ring = " << ringNumber 
		<< " Stereo = " << stereo
		<< std::endl;
      
      std::cout << "Track/z/r : "
		<< simTrackId << " " 
		<< geomDet->surface().toGlobal(iterRecHit->localPosition()).z() << " " 
		<< geomDet->surface().toGlobal(iterRecHit->localPosition()).perp() << std::endl;
#endif
    
    }

    //
    // Create a starting trajectory with SimTrack + first RecHit
    
    // take the SimTrack parameters

#ifdef FAMOS_DEBUG
    std::cout << " SimTrack = " <<  (*theSimTracks)[simTrackId]
	      << "\tVERT ind = " << vertexIndex 
	      << " (x,y,z)= (" <<  (*theSimVtx)[vertexIndex].position() 
	      << ")" << std::endl;
#endif
    
    GlobalPoint  position((*theSimVtx)[vertexIndex].position().x(),
			  (*theSimVtx)[vertexIndex].position().y(),
			  (*theSimVtx)[vertexIndex].position().z());
    
    
    GlobalVector momentum( (*theSimTracks)[simTrackId].momentum().x() , 
			   (*theSimTracks)[simTrackId].momentum().y() , 
			   (*theSimTracks)[simTrackId].momentum().z() );

    float        charge   = (*theSimTracks)[simTrackId].charge();

    GlobalTrajectoryParameters initialParams(position,momentum,(int)charge,theMagField);

    AlgebraicSymMatrix errorMatrix(5,1);

    //why?
    //    errorMatrix = errorMatrix * 10;
    // Ben oui, pourquoi?
    
#ifdef FAMOS_DEBUG
    std::cout << "GSTrackCandidateMaker: SimTrack parameters " << std::endl;
    std::cout << "\t\t pT  = " << (*theSimTracks)[simTrackId].momentum().perp() << std::endl;
    std::cout << "\t\t eta = " << (*theSimTracks)[simTrackId].momentum().eta()  << std::endl;
    std::cout << "\t\t phi = " << (*theSimTracks)[simTrackId].momentum().phi()  << std::endl;
    std::cout << "GSTrackCandidateMaker: AlgebraicSymMatrix " << errorMatrix << std::endl;
#endif
    
    // Construct TSOS from FTS + Surface
    CurvilinearTrajectoryError initialError(errorMatrix);
    FreeTrajectoryState initialFTS(initialParams, initialError);
    
#ifdef FAMOS_DEBUG
    std::cout << "GSTrackCandidateMaker: FTS momentum " << initialFTS.momentum() << std::endl;
#endif
    
    const GeomDetUnit* initialLayer = theGeometry->idToDetUnit( recHits.front().geographicalId() );
    const TrajectoryStateOnSurface initialTSOS(initialFTS, initialLayer->surface());

#ifdef FAMOS_DEBUG
    std::cout << "GSTrackCandidateMaker: TSOS global momentum "    << initialTSOS.globalMomentum() << std::endl;
    std::cout << "\t\t\tpT = "                                     << initialTSOS.globalMomentum().perp() << std::endl;
    std::cout << "\t\t\teta = "                                    << initialTSOS.globalMomentum().eta() << std::endl;
    std::cout << "\t\t\tphi = "                                    << initialTSOS.globalMomentum().phi() << std::endl;
    std::cout << "GSTrackCandidateMaker: TSOS local momentum "     << initialTSOS.localMomentum()  << std::endl;
    std::cout << "GSTrackCandidateMaker: TSOS local error "        << initialTSOS.localError().positionError() << std::endl;
    std::cout << "GSTrackCandidateMaker: TSOS local error matrix " << initialTSOS.localError().matrix() << std::endl;
    std::cout << "GSTrackCandidateMaker: TSOS surface side "       << initialTSOS.surfaceSide()    << std::endl;
    //    std::cout << "GSTrackCandidateMaker: TSOS X0[cm] = "           << initialTSOS.surface().mediumProperties()->radLen() << std::endl;
#endif
    // This new method is here to avod a memory leak that came with 
    // TrackingTools/TrajectoryState/src/TrajectoryStateTransform.cc
    // from the method persistentState().
    stateOnDet(initialTSOS, 
	       recHits.front().geographicalId().rawId(),
	       initialState);

#ifdef FAMOS_DEBUG
    std::cout << "GSTrackCandidateMaker: detid " << recHits.front().geographicalId().rawId() << std::endl;
    std::cout << "GSTrackCandidateMaker: PTSOS detId " << initialState.detId() << std::endl;
    std::cout << "GSTrackCandidateMaker: PTSOS local momentum " << initialState.parameters().momentum() << std::endl;
#endif
    //
    
    // Track Candidate stored
    TrackCandidate newTrackCandidate(recHits, TrajectorySeed(initialState, recHits, alongMomentum), initialState );
    
#ifdef FAMOS_DEBUG
    // Log
    std::cout << "\tSeed Information " << std::endl;
    std::cout << "\tSeed Direction = " << TrajectorySeed(initialState, recHits, alongMomentum).direction() << std::endl;
    std::cout << "\tSeed StartingDet = " << TrajectorySeed(initialState, recHits, alongMomentum).startingState().detId() << std::endl;
    
    std::cout << "\tTrajectory Parameters " 
	      << std::endl;
    std::cout << "\t\t detId  = " 
	      << newTrackCandidate.trajectoryStateOnDet().detId() 
	      << std::endl;
    std::cout << "\t\t loc.px = " 
	      << newTrackCandidate.trajectoryStateOnDet().parameters().momentum().x()    
	      << std::endl;
    std::cout << "\t\t loc.py = " 
	      << newTrackCandidate.trajectoryStateOnDet().parameters().momentum().y()    
	      << std::endl;
    std::cout << "\t\t loc.pz = " 
	      << newTrackCandidate.trajectoryStateOnDet().parameters().momentum().z()    
	      << std::endl;
    std::cout << "\t\t error  = ";
    for(std::vector< float >::const_iterator iElement = newTrackCandidate.trajectoryStateOnDet().errorMatrix().begin();
	iElement < newTrackCandidate.trajectoryStateOnDet().errorMatrix().end();
	++iElement) {
      std::cout << "\t" << *iElement;
    }
    std::cout << std::endl;
#endif

    output->push_back(newTrackCandidate);
    ++nTrackCandidates;

  }
  
#ifdef FAMOS_DEBUG
  std::cout << " GSTrackCandidateMaker: Total SimTracks           = " << nSimTracks << std::endl 
	    << "                        Total SimTracksWithHits   = " << nTracksWithHits  << std::endl 
	    << "                        Total SimTracksWithPT     = " << nTracksWithPT  << std::endl 
	    << "                        Total SimTracksWithD0Z0   = " << nTracksWithD0Z0  << std::endl 
	    << "                        Total Track Candidates    = " << nTrackCandidates 
	    << std::endl;
#endif
  
  e.put(output);

}


// This is a copy of a method in 
// TrackingTools/TrajectoryState/src/TrajectoryStateTransform.cc
// but it does not return a pointer (thus avoiding a memory leak)
// In addition, it's also CPU more efficient, because 
// ts.localError().matrix() is not copied
void 
GSTrackCandidateMaker::stateOnDet(const TrajectoryStateOnSurface& ts,
				  unsigned int detid,
				  PTrajectoryStateOnDet& pts) const
{

  const AlgebraicSymMatrix55& m = ts.localError().matrix();
  
  int dim = 5; /// should check if corresponds to m

  float localErrors[15];
  int k = 0;
  for (int i=0; i<dim; ++i) {
    for (int j=0; j<=i; ++j) {
      localErrors[k++] = m(i,j);
    }
  }
  int surfaceSide = static_cast<int>(ts.surfaceSide());

  pts = PTrajectoryStateOnDet( ts.localParameters(),
			       localErrors, detid,
			       surfaceSide);
}


bool
GSTrackCandidateMaker::compatibleWithVertex(GlobalPoint& gpos1, GlobalPoint& gpos2) {


 // The hits 1 and 2 positions, in HepLorentzVector's
  XYZTLorentzVector thePos1(gpos1.x(),
			   gpos1.y(),
			   gpos1.z(),
			   0.);

  XYZTLorentzVector thePos2(gpos2.x(),
			   gpos2.y(),
			   gpos2.z(),
			   0.);

  // Create new particles that pass through the second hit with pT = ptMin 
  // and charge = +/-1
  XYZTLorentzVector theMom2 = (thePos2-thePos1);

  theMom2 /= theMom2.Pt();
  theMom2.SetE(sqrt(theMom2.Vect().Mag2()));

  // The corresponding RawParticles (to be propagated) for e- and e+
  ParticlePropagator myElecL(theMom2,thePos2,-1.);
  ParticlePropagator myPosiL(theMom2,thePos2,+1.);


  // Propagate to the closest approach point, with the constraint that 
  // the particles should pass through the  first hit
  myElecL.propagateToNominalVertex(thePos1);
  myPosiL.propagateToNominalVertex(thePos1);

  theMom2 *= 1000.0;//ptmax
  // The corresponding RawParticles (to be propagated) for e- and e+
  ParticlePropagator myElecH(theMom2,thePos2,-1.);
  ParticlePropagator myPosiH(theMom2,thePos2,+1.);

  // Propagate to the closest approach point, with the constraint that 
  // the particles should pass through the  first hit
  myElecH.propagateToNominalVertex(thePos1);
  myPosiH.propagateToNominalVertex(thePos1);

  // And check at least one of the particles statisfy the SeedGenerator
  // constraint (originRadius, originHalfLength)

  double originRadius = 0.2;
  double originHalfLength = 15.;

  /*
  std::cout << " Neg Charge L R = " << myElecL.R() << "\t Z = " << fabs(myElecL.Z()) << std::endl;
  std::cout << " Pos Charge L R = " << myPosiL.R() << "\t Z = " << fabs(myPosiL.Z()) << std::endl;
  std::cout << " Neg Charge H R = " << myElecH.R() << "\t Z = " << fabs(myElecH.Z()) << std::endl;
  std::cout << " Pos Charge H R = " << myPosiH.R() << "\t Z = " << fabs(myPosiH.Z()) << std::endl;
  */

  if ( myElecL.R() < originRadius && 
       fabs(myElecL.Z()) < originHalfLength ) return true;
  if ( myPosiL.R() < originRadius && 
       fabs(myPosiL.Z()) < originHalfLength ) return true;
  if ( myElecH.R() < originRadius && 
       fabs(myElecH.Z()) < originHalfLength ) return true;
  if ( myPosiH.R() < originRadius && 
       fabs(myPosiH.Z()) < originHalfLength ) return true;

  return false;
}
