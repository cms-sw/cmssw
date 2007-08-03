#include <memory>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h" 
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "FastSimulation/Tracking/interface/GSTrackCandidateMaker.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h" 
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h" 
#include "DataFormats/SiStripDetId/interface/TECDetId.h" 

#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "FastSimulation/ParticlePropagator/interface/ParticlePropagator.h"
//

//for debug only 
//#define FAMOS_DEBUG

GSTrackCandidateMaker::GSTrackCandidateMaker(const edm::ParameterSet& conf) 
{  
#ifdef FAMOS_DEBUG
  std::cout << "GSTrackCandidateMaker created" << std::endl;
#endif
  produces<TrackCandidateCollection>();
  
  // The smallest true pT for a track candidate
  pTMin = conf.getParameter<double>("pTMin");
  pTMin *= pTMin;  // Cut is done of perp2() - CPU saver
  
  // The smallest number of Rec Hits for a track candidate
  minRecHits = conf.getParameter<unsigned int>("MinRecHits");

  // The smallest true impact parameters (d0 and z0) for a track candidate
  maxD0 = conf.getParameter<double>("MaxD0");
  maxZ0 = conf.getParameter<double>("MaxZ0");

  // The name of the hit producer
  hitProducer = conf.getParameter<std::string>("HitProducer");

  // The cuts for seed cleaning
  seedCleaning = conf.getParameter<bool>("seedCleaning");
  std::string seedingType = conf.getParameter<std::string>("seedType");
  if ( seedingType == "GlobalPixel" ) seedType = 1;
  else if ( seedingType == "GlobalMixed" ) seedType = 2;
  else seedCleaning = false;
  originRadius = conf.getParameter<double>("originRadius");
  originHalfLength = conf.getParameter<double>("originHalfLength");
  originpTMin = conf.getParameter<double>("originpTMin");

  // Reject overlapping hits?
  rejectOverlaps = conf.getParameter<bool>("overlapCleaning");

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
  e.getByLabel("famosSimHits",theSimTracks);
  
  edm::Handle<edm::SimVertexContainer> theSimVtx;
  e.getByLabel("famosSimHits",theSimVtx);

#ifdef FAMOS_DEBUG
  std::cout << " Step A: SimTracks found " << theSimTracks->size() << std::endl;
#endif
  
  edm::Handle<SiTrackerGSRecHit2DCollection> theGSRecHits;
  e.getByLabel(hitProducer, theGSRecHits);
  
  // No tracking attempted if no hits (but put an empty collection in the event)!
#ifdef FAMOS_DEBUG
  std::cout << " Step B: Full GS RecHits found " << theGSRecHits->size() << std::endl;
#endif
  if(theGSRecHits->size() == 0) {
    e.put(output);
    return;
  }
  
  
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
    
    if ( seedType == 1 ) {
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
	  GlobalPoint gpos1 = geomDet->surface().toGlobal(hit1->localPosition());
	  for ( iterRecHit2 = iterRecHit+1; iterRecHit2 != theRecHitRangeIteratorEnd; ++iterRecHit2) {
	    hit2 = &(*iterRecHit2);
	    if((unsigned int)hit2->geographicalId().subdetId()== PixelSubdetector::PixelBarrel || 
	       (unsigned int)hit2->geographicalId().subdetId()== PixelSubdetector::PixelEndcap){
	      const DetId& detId = hit2->geographicalId();
	      const GeomDet* geomDet( theGeometry->idToDet(detId) );
	      GlobalPoint gpos2 = geomDet->surface().toGlobal(hit2->localPosition());
	      
	      compatible = compatibleWithVertex(gpos1,gpos2);
	      
	      if(compatible) break;
	    }
	  }
	  if(compatible) break;
	}
      }
      if(!compatible) continue;

    } else if ( seedType == 2 ) {

      bool compatible = false;
      const SiTrackerGSRecHit2D *hit1;
      const SiTrackerGSRecHit2D *hit2;
      for ( iterRecHit = theRecHitRangeIteratorBegin; 
	    iterRecHit != theRecHitRangeIteratorEnd; 
	    ++iterRecHit) {
	hit1 = &(*iterRecHit);
	if((unsigned int)(hit1->geographicalId().subdetId())== PixelSubdetector::PixelBarrel || 
	   (unsigned int)(hit1->geographicalId().subdetId())== PixelSubdetector::PixelEndcap ) {
	  const DetId& detId = hit1->geographicalId();
	  const GeomDet* geomDet( theGeometry->idToDet(detId) );
	  GlobalPoint gpos1 = geomDet->surface().toGlobal(hit1->localPosition());
	  
	  for ( iterRecHit2 = iterRecHit+1; iterRecHit2 != theRecHitRangeIteratorEnd; ++iterRecHit2) {
	    hit2 = &(*iterRecHit2);
	    const DetId& detId = hit2->geographicalId();
	    const GeomDet* geomDet( theGeometry->idToDet(detId) );
	    GlobalPoint gpos2 = geomDet->surface().toGlobal(hit2->localPosition());

	    if((unsigned int)hit2->geographicalId().subdetId()== PixelSubdetector::PixelBarrel || 
	       (unsigned int)hit2->geographicalId().subdetId()== PixelSubdetector::PixelEndcap){

	      compatible = compatibleWithVertex(gpos1,gpos2);

	      if(compatible) break;
	      
	    } else if ( (unsigned int)(hit1->geographicalId().subdetId())== PixelSubdetector::PixelEndcap){
	      PXFDetId id1(hit1->geographicalId().rawId());
	      if(id1.disk()==2){
		if ((unsigned int)(hit2->geographicalId().subdetId())== StripSubdetector::TEC) {
		  TECDetId id2(hit2->geographicalId().rawId());
		  if(id2.wheel()<4 && id2.ring()<3){
		    
		    compatible = compatibleWithVertex(gpos1,gpos2);
		    //std::cout << "Pair into PXF and TEC = " << id2.wheel() << ",\t" << id2.ring() << std::endl; 
		    if(compatible) break;
		  }
		}
	      }
	    }//rh2 condition
	  }//rh2
	  if(compatible) break;
	}//rh1 condition
      }//rh1	  
      if(!compatible) continue;
    }
	  
    
    // Create OwnVector with sorted GSRecHit's
    unsigned int previousLayerNumber = 0;
    unsigned int previousSubdetId = 0;
    double previousLambda1 = 0.;
    bool replacePreviousHit = false;
    bool keepThisHit = true;
    TrackingRecHit* previousHit = 0;
    //
    edm::OwnVector<TrackingRecHit> recHits;
    // 
    for ( iterRecHit = theRecHitRangeIteratorBegin; 
	   iterRecHit != theRecHitRangeIteratorEnd; 
	   ++iterRecHit) {
      const DetId& detId =  iterRecHit->geographicalId();

      if ( rejectOverlaps ) { 

	unsigned int subdetId = detId.subdetId(); 
	unsigned int layerNumber=0;
	unsigned int ringNumber = 0;
	unsigned int stereo = 0;
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

	// Determine the smaller uncertainty for the current hit
	double xx = iterRecHit->localPositionError().xx();
	double yy = iterRecHit->localPositionError().yy();
	double xy = iterRecHit->localPositionError().xy();
	double delta = std::sqrt((xx-yy)*(xx-yy)+4.*xy*xy);
	double lambda1 = 0.5 * (xx+yy-delta);

	// Not the same layer : Add the current hit
	if ( subdetId != previousSubdetId || layerNumber != previousLayerNumber ) {
	  keepThisHit = true;
	  replacePreviousHit = false;
	// Same layer : keep the better hit, and drop the other 
	} else {
	  // In that case, the current hit is better -> drop the previous one
	  if ( lambda1 < previousLambda1 ) {
	    keepThisHit = true;
	    replacePreviousHit = true;
	  // In this case, the previous hit was better -> do nothing ! (ignore the current hit)
	  } else {
	    keepThisHit = false;
	    replacePreviousHit = false;
	  }
	}
	previousSubdetId = subdetId;
	previousLayerNumber = layerNumber;
	previousLambda1 = lambda1;
      }

      // delete previous hit
      if ( replacePreviousHit ) {
	delete previousHit;
	recHits.pop_back();
#ifdef FAMOS_DEBUG
	std::cout << "Removed this RecHit (found a better hit on the same layer)" << std::endl; 
#endif
      }

      // add current hit
      if ( keepThisHit ) { 

	const GeomDet* geomDet( theGeometry->idToDet(detId) );
	TrackingRecHit* aTrackingRecHit = 
	  GenericTransientTrackingRecHit::build(geomDet,&(*iterRecHit))->hit()->clone();
	recHits.push_back(aTrackingRecHit);	
	previousHit = aTrackingRecHit;

#ifdef FAMOS_DEBUG
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
    
      // ignore current hit
      } else { 
	previousHit = 0;
      }
      
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
    errorMatrix = errorMatrix * 10;
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
    
    // Track Candidate stored if the number of hits is large enough.
    if ( recHits.size() < minRecHits ) continue;
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

  if ( !seedCleaning ) return true;

  // The hits 1 and 2 positions, in HepLorentzVector's
  XYZTLorentzVector thePos1(gpos1.x(),gpos1.y(),gpos1.z(),0.);
  XYZTLorentzVector thePos2(gpos2.x(),gpos2.y(),gpos2.z(),0.);

  // Create new particles that pass through the second hit with pT = ptMin 
  // and charge = +/-1
  XYZTLorentzVector theMom2 = (thePos2-thePos1);

  theMom2 /= theMom2.Pt();
  theMom2 *= originpTMin;
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
