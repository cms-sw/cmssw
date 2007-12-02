#include <memory>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OwnVector.h"
//#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h" 
//#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "FastSimulation/ParticlePropagator/interface/MagneticFieldMapRecord.h"

#include "FastSimulation/Tracking/plugins/TrajectorySeedProducer.h"
#include "FastSimulation/Tracking/interface/TrackerRecHit.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"
/*
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h" 
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h" 
#include "DataFormats/SiStripDetId/interface/TECDetId.h" 
*/


#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "FastSimulation/ParticlePropagator/interface/ParticlePropagator.h"
#include "FastSimulation/ParticlePropagator/interface/MagneticFieldMap.h"
//

//for debug only 
#define FAMOS_DEBUG

TrajectorySeedProducer::TrajectorySeedProducer(const edm::ParameterSet& conf) 
{  
  std::cout << "TrajectorySeedProducer initializing" << std::endl;
  
  // The name of the TrajectorySeed Collections
  seedingAlgo = conf.getParameter<std::vector<std::string> >("seedingAlgo");
  for ( unsigned i=0; i<seedingAlgo.size(); ++i )
    produces<TrajectorySeedCollection>(seedingAlgo[i]);

  // The smallest true pT for a track candidate
  pTMin = conf.getParameter<std::vector<double> >("pTMin");
  if ( pTMin.size() != seedingAlgo.size() ) 
    throw cms::Exception("FastSimulation/TrajectorySeedProducer ") 
      << " WARNING : pTMin does not have the proper size "
      << std::endl;

  for ( unsigned i=0; i<pTMin.size(); ++i )
    pTMin[i] *= pTMin[i];  // Cut is done of perp2() - CPU saver
  
  // The smallest number of Rec Hits for a track candidate
  minRecHits = conf.getParameter<std::vector<unsigned int> >("minRecHits");
  if ( minRecHits.size() != seedingAlgo.size() ) 
    throw cms::Exception("FastSimulation/TrajectorySeedProducer ") 
      << " WARNING : minRecHits does not have the proper size "
      << std::endl;

  // The smallest true impact parameters (d0 and z0) for a track candidate
  maxD0 = conf.getParameter<std::vector<double> >("maxD0");
  if ( maxD0.size() != seedingAlgo.size() ) 
    throw cms::Exception("FastSimulation/TrajectorySeedProducer ") 
      << " WARNING : maxD0 does not have the proper size "
      << std::endl;

  maxZ0 = conf.getParameter<std::vector<double> >("maxZ0");
  if ( maxZ0.size() != seedingAlgo.size() ) 
    throw cms::Exception("FastSimulation/TrajectorySeedProducer ") 
      << " WARNING : maxZ0 does not have the proper size "
      << std::endl;

  // The name of the hit producer
  hitProducer = conf.getParameter<edm::InputTag>("HitProducer");

  // The cuts for seed cleaning
  seedCleaning = conf.getParameter<bool>("seedCleaning");

  // Number of hits needed for a seed
  numberOfHits = conf.getParameter<std::vector<unsigned int> >("numberOfHits");
  if ( numberOfHits.size() != seedingAlgo.size() ) 
    throw cms::Exception("FastSimulation/TrajectorySeedProducer ") 
      << " WARNING : numberOfHits does not have the proper size "
      << std::endl;

  //
  firstHitSubDetectorNumber = 
    conf.getParameter<std::vector<unsigned int> >("firstHitSubDetectorNumber");
  if ( firstHitSubDetectorNumber.size() != seedingAlgo.size() ) 
    throw cms::Exception("FastSimulation/TrajectorySeedProducer ") 
      << " WARNING : firstHitSubDetectorNumber does not have the proper size "
      << std::endl;

  std::vector<unsigned int> firstSubDets = 
    conf.getParameter<std::vector<unsigned int> >("firstHitSubDetectors");
  unsigned isub1 = 0;
  unsigned check1 = 0;
  firstHitSubDetectors.resize(seedingAlgo.size());
  for ( unsigned ialgo=0; ialgo<firstHitSubDetectorNumber.size(); ++ialgo ) { 
    check1 += firstHitSubDetectorNumber[ialgo];
    for ( unsigned idet=0; idet<firstHitSubDetectorNumber[ialgo]; ++idet ) { 
      firstHitSubDetectors[ialgo].push_back(firstSubDets[isub1++]);
    }
  }
  if ( firstSubDets.size() != check1 ) 
    throw cms::Exception("FastSimulation/TrajectorySeedProducer ") 
      << " WARNING : firstHitSubDetectors does not have the proper size (should be " << check1 << ")"
      << std::endl;


  secondHitSubDetectorNumber = 
    conf.getParameter<std::vector<unsigned int> >("secondHitSubDetectorNumber");
  if ( secondHitSubDetectorNumber.size() != seedingAlgo.size() ) 
    throw cms::Exception("FastSimulation/TrajectorySeedProducer ") 
      << " WARNING : secondHitSubDetectorNumber does not have the proper size "
      << std::endl;

  std::vector<unsigned int> secondSubDets = 
    conf.getParameter<std::vector<unsigned int> >("secondHitSubDetectors");
  unsigned isub2 = 0;
  unsigned check2 = 0;
  secondHitSubDetectors.resize(seedingAlgo.size());
  for ( unsigned ialgo=0; ialgo<secondHitSubDetectorNumber.size(); ++ialgo ) { 
    check2 += secondHitSubDetectorNumber[ialgo];
    for ( unsigned idet=0; idet<secondHitSubDetectorNumber[ialgo]; ++idet ) { 
      secondHitSubDetectors[ialgo].push_back(secondSubDets[isub2++]);
    }
  }
  if ( secondSubDets.size() != check2 ) 
    throw cms::Exception("FastSimulation/TrajectorySeedProducer ") 
      << " WARNING : secondHitSubDetectors does not have the proper size (should be " << check2 << ")"
      << std::endl;

  thirdHitSubDetectorNumber = 
    conf.getParameter<std::vector<unsigned int> >("thirdHitSubDetectorNumber");
  if ( thirdHitSubDetectorNumber.size() != seedingAlgo.size() ) 
    throw cms::Exception("FastSimulation/TrajectorySeedProducer ") 
      << " WARNING : thirdHitSubDetectorNumber does not have the proper size "
      << std::endl;

  std::vector<unsigned int> thirdSubDets = 
    conf.getParameter<std::vector<unsigned int> >("thirdHitSubDetectors");
  unsigned isub3 = 0;
  unsigned check3 = 0;
  thirdHitSubDetectors.resize(seedingAlgo.size());
  for ( unsigned ialgo=0; ialgo<thirdHitSubDetectorNumber.size(); ++ialgo ) { 
    check3 += thirdHitSubDetectorNumber[ialgo];
    for ( unsigned idet=0; idet<thirdHitSubDetectorNumber[ialgo]; ++idet ) { 
      thirdHitSubDetectors[ialgo].push_back(thirdSubDets[isub3++]);
    }
  }
  if ( thirdSubDets.size() != check3 ) 
    throw cms::Exception("FastSimulation/TrajectorySeedProducer ") 
      << " WARNING : thirdHitSubDetectors does not have the proper size (should be " << check3 << ")"
      << std::endl;

  originRadius = conf.getParameter<std::vector<double> >("originRadius");
  if ( originRadius.size() != seedingAlgo.size() ) 
    throw cms::Exception("FastSimulation/TrajectorySeedProducer ") 
      << " WARNING : originRadius does not have the proper size "
      << std::endl;

  originHalfLength = conf.getParameter<std::vector<double> >("originHalfLength");
  if ( originHalfLength.size() != seedingAlgo.size() ) 
    throw cms::Exception("FastSimulation/TrajectorySeedProducer ") 
      << " WARNING : originHalfLength does not have the proper size "
      << std::endl;

  originpTMin = conf.getParameter<std::vector<double> >("originpTMin");
  if ( originpTMin.size() != seedingAlgo.size() ) 
    throw cms::Exception("FastSimulation/TrajectorySeedProducer ") 
      << " WARNING : originpTMin does not have the proper size "
      << std::endl;


  // Reject overlapping hits?
  rejectOverlaps = conf.getParameter<bool>("overlapCleaning");

}

  
// Virtual destructor needed.
TrajectorySeedProducer::~TrajectorySeedProducer() {

  // do nothing
#ifdef FAMOS_DEBUG
  std::cout << "TrajectorySeedProducer destructed" << std::endl;
#endif

} 
 
void 
TrajectorySeedProducer::beginJob (edm::EventSetup const & es) {

  //services
  //  es.get<TrackerRecoGeometryRecord>().get(theGeomSearchTracker);

  edm::ESHandle<MagneticField>          magField;
  edm::ESHandle<TrackerGeometry>        geometry;
  edm::ESHandle<MagneticFieldMap>       magFieldMap;


  es.get<IdealMagneticFieldRecord>().get(magField);
  es.get<TrackerDigiGeometryRecord>().get(geometry);
  es.get<MagneticFieldMapRecord>().get(magFieldMap);

  theMagField = &(*magField);
  theGeometry = &(*geometry);
  theFieldMap = &(*magFieldMap);

  const GlobalPoint g(0.,0.,0.);

}
  
  // Functions that gets called by framework every event
void 
TrajectorySeedProducer::produce(edm::Event& e, const edm::EventSetup& es) {        

#ifdef FAMOS_DEBUG
  std::cout << "################################################################" << std::endl;
  std::cout << " TrajectorySeedProducer produce init " << std::endl;
#endif

  unsigned nSimTracks = 0;
  unsigned nTracksWithHits = 0;
  unsigned nTracksWithPT = 0;
  unsigned nTracksWithD0Z0 = 0;
  //  unsigned nTrackCandidates = 0;
  PTrajectoryStateOnDet initialState;
  
  // Output
  std::vector<TrajectorySeedCollection*>
    output(seedingAlgo.size(),static_cast<TrajectorySeedCollection*>(0));
  for ( unsigned ialgo=0; ialgo<seedingAlgo.size(); ++ialgo ) { 
    //    std::auto_ptr<TrajectorySeedCollection> p(new TrajectorySeedCollection );
    output[ialgo] = new TrajectorySeedCollection;
  }
  
  // Input
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
    for ( unsigned ialgo=0; ialgo<seedingAlgo.size(); ++ialgo ) {
      std::auto_ptr<TrajectorySeedCollection> p(output[ialgo]);
      e.put(p,seedingAlgo[ialgo]);
    }
    return;
  }
  
  
#ifdef FAMOS_DEBUG
  std::cout << " Step C: Loop over the RecHits, track by track " << std::endl;
#endif

  // The vector of simTrack Id's carrying GSRecHits
  const std::vector<unsigned> theSimTrackIds = theGSRecHits->ids();

  // loop over SimTrack Id's
  for ( unsigned tkId=0;  tkId != theSimTrackIds.size(); ++tkId ) {

    std::cout << "Track number " << tkId << std::endl;

    ++nSimTracks;
    unsigned simTrackId = theSimTrackIds[tkId];
    const SimTrack& theSimTrack = (*theSimTracks)[simTrackId]; 
    std::cout << "Pt = " << std::sqrt(theSimTrack.momentum().Perp2()) 
	      << " eta " << theSimTrack.momentum().Eta()
	      << std::endl;

    SiTrackerGSRecHit2DCollection::range theRecHitRange = theGSRecHits->get(simTrackId);
    SiTrackerGSRecHit2DCollection::const_iterator theRecHitRangeIteratorBegin = theRecHitRange.first;
    SiTrackerGSRecHit2DCollection::const_iterator theRecHitRangeIteratorEnd   = theRecHitRange.second;
    SiTrackerGSRecHit2DCollection::const_iterator iterRecHit;
    SiTrackerGSRecHit2DCollection::const_iterator iterRecHit1;
    SiTrackerGSRecHit2DCollection::const_iterator iterRecHit2;
    SiTrackerGSRecHit2DCollection::const_iterator iterRecHit3;

    // Loop on the successive seedings
    for ( unsigned int ialgo = 0; ialgo < seedingAlgo.size(); ++ialgo ) { 

      std::cout << "Algo " << seedingAlgo[ialgo] << std::endl;

      // Request a minimum number of RecHits for the track to give a seed.
      unsigned numberOfRecHits = 0;
      for ( iterRecHit = theRecHitRangeIteratorBegin; 
	    iterRecHit != theRecHitRangeIteratorEnd; 
	    ++iterRecHit) ++numberOfRecHits;
      std::cout << "The number of RecHits = " << numberOfRecHits << std::endl;
      if ( numberOfRecHits < minRecHits[ialgo] ) continue;
      ++nTracksWithHits;

      // Request a minimum pT for the sim track
      if ( theSimTrack.momentum().Perp2() < pTMin[ialgo] ) continue;
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
      if ( theParticle.xyImpactParameter() > maxD0[ialgo] ) continue;
      if ( fabs( theParticle.zImpactParameter() ) > maxZ0[ialgo] ) continue;
      ++nTracksWithD0Z0;
      
      std::vector<TrackerRecHit> theSeedHits(numberOfHits[ialgo],static_cast<TrackerRecHit>(TrackerRecHit()));
      bool compatible = false;
      for ( iterRecHit1 = theRecHitRangeIteratorBegin; iterRecHit1 != theRecHitRangeIteratorEnd; ++iterRecHit1) {
	theSeedHits[0] = TrackerRecHit(&(*iterRecHit1),theGeometry);
	std::cout << "The first hit position = " << theSeedHits[0].globalPosition() << std::endl;
	std::cout << "The first hit subDetId = " << theSeedHits[0].subDetId() << std::endl;
	std::cout << "The first hit layer    = " << theSeedHits[0].layerNumber() << std::endl;
	// if ( !theSeedHits[0].isOnRequestedDet(firstHitSubDetectors[ialgo]) ) continue;
	if ( !theSeedHits[0].isOnRequestedDet(firstHitSubDetectors[ialgo]) ) break;
	std::cout << "Apparently the first hit is on the requested detector! " << std::endl;
	GlobalPoint gpos1 = theSeedHits[0].globalPosition();
	for ( iterRecHit2 = iterRecHit1+1; iterRecHit2 != theRecHitRangeIteratorEnd; ++iterRecHit2) {
	  theSeedHits[1] = TrackerRecHit(&(*iterRecHit2),theGeometry);
	  std::cout << "The second hit position = " << theSeedHits[1].globalPosition() << std::endl;
	  std::cout << "The second hit subDetId = " << theSeedHits[1].subDetId() << std::endl;
	  std::cout << "The second hit layer    = " << theSeedHits[1].layerNumber() << std::endl;

	  if ( !theSeedHits[1].isOnRequestedDet(firstHitSubDetectors[ialgo]) ) break;

	  if ( !( theSeedHits[1].isOnRequestedDet(secondHitSubDetectors[ialgo]) && 
		( theSeedHits[1].subDetId() > theSeedHits[0].subDetId() || 
		  theSeedHits[1].layerNumber() > theSeedHits[0].layerNumber() ) ) ) continue;
	  std::cout << "Apparently the first hit is on the requested detector! " << std::endl;
	  GlobalPoint gpos2 = theSeedHits[1].globalPosition();
	  compatible = compatibleWithVertex(gpos1,gpos2,ialgo);
	  std::cout << "Are the two hits compatible with the PV? " << compatible << std::endl;
	  if ( !compatible ) continue;
	  std::cout << "Pair kept! " << std::endl;
	  if ( numberOfHits[ialgo] == 2 ) break; 
	  for ( iterRecHit3 = iterRecHit2+1; iterRecHit3 != theRecHitRangeIteratorEnd; ++iterRecHit3) {
	    theSeedHits[2] = TrackerRecHit(&(*iterRecHit3),theGeometry);

	    if ( !theSeedHits[2].isOnRequestedDet(firstHitSubDetectors[ialgo]) ) break;

	    compatible = theSeedHits[2].isOnRequestedDet(thirdHitSubDetectors[ialgo]) && 
		       ( theSeedHits[2].subDetId() > theSeedHits[1].subDetId() || 
			 theSeedHits[2].layerNumber() > theSeedHits[1].layerNumber() );
	    if ( compatible ) break;	  
	  }
	  if ( compatible ) break;
	}
	if ( compatible ) break;
      }

      // There is no compatible seed for this track with this seeding algorithm 
      // Go to next algo
      if ( !compatible ) continue;

      std::cout << "Preparing to create the TrajectorySeed" << std::endl;
      // The seed is validated -> include in the collection
      // 1) Create the vector of RecHits
      edm::OwnVector<TrackingRecHit> recHits;
      for ( unsigned ih=0; ih<theSeedHits.size(); ++ih ) { 
	TrackingRecHit* aTrackingRecHit = 
	  GenericTransientTrackingRecHit::build(theSeedHits[ih].geomDet(),
						theSeedHits[ih].hit())->hit()->clone();
	recHits.push_back(aTrackingRecHit);
      }
      std::cout << "with " << recHits.size() << " hits." << std::endl;

      // 2) Create the initial state
      //   a) origin vertex
      GlobalPoint  position((*theSimVtx)[vertexIndex].position().x(),
			    (*theSimVtx)[vertexIndex].position().y(),
			    (*theSimVtx)[vertexIndex].position().z());
      
      //   b) initial momentum
      GlobalVector momentum( (*theSimTracks)[simTrackId].momentum().x() , 
			     (*theSimTracks)[simTrackId].momentum().y() , 
			     (*theSimTracks)[simTrackId].momentum().z() );
      //   c) electric charge
      float        charge   = (*theSimTracks)[simTrackId].charge();
      //  -> inital parameters
      GlobalTrajectoryParameters initialParams(position,momentum,(int)charge,theMagField);
      //  -> large initial errors
      AlgebraicSymMatrix errorMatrix(5,1);      
      errorMatrix = errorMatrix * 10;
      CurvilinearTrajectoryError initialError(errorMatrix);
      // -> initial state
      FreeTrajectoryState initialFTS(initialParams, initialError);      
      const GeomDetUnit* initialLayer = theGeometry->idToDetUnit( recHits.front().geographicalId() );
      const TrajectoryStateOnSurface initialTSOS(initialFTS, initialLayer->surface());      
      stateOnDet(initialTSOS, 
		 recHits.front().geographicalId().rawId(),
		 initialState);
      // Create a new Trajectory Seed    
      output[ialgo]->push_back(TrajectorySeed(initialState, recHits, alongMomentum));
      std::cout << "Trajectory seed created ! " << std::endl;
      break;
      // End of the loop over seeding algorithms
    }
    // End on the loop over simtracks
  }

  for ( unsigned ialgo=0; ialgo<seedingAlgo.size(); ++ialgo ) { 
    std::auto_ptr<TrajectorySeedCollection> p(output[ialgo]);
    e.put(p,seedingAlgo[ialgo]);
  }

}

// This is a copy of a method in 
// TrackingTools/TrajectoryState/src/TrajectoryStateTransform.cc
// but it does not return a pointer (thus avoiding a memory leak)
// In addition, it's also CPU more efficient, because 
// ts.localError().matrix() is not copied
void 
TrajectorySeedProducer::stateOnDet(const TrajectoryStateOnSurface& ts,
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
TrajectorySeedProducer::compatibleWithVertex(GlobalPoint& gpos1, 
					     GlobalPoint& gpos2,
					     unsigned algo) const {

  if ( !seedCleaning ) return true;

  // The hits 1 and 2 positions, in HepLorentzVector's
  XYZTLorentzVector thePos1(gpos1.x(),gpos1.y(),gpos1.z(),0.);
  XYZTLorentzVector thePos2(gpos2.x(),gpos2.y(),gpos2.z(),0.);

  // Create new particles that pass through the second hit with pT = ptMin 
  // and charge = +/-1
  XYZTLorentzVector theMom2 = (thePos2-thePos1);

  theMom2 /= theMom2.Pt();
  theMom2 *= originpTMin[algo];
  theMom2.SetE(sqrt(theMom2.Vect().Mag2()));

  // The corresponding RawParticles (to be propagated) for e- and e+
  ParticlePropagator myElecL(theMom2,thePos2,-1.,theFieldMap);
  ParticlePropagator myPosiL(theMom2,thePos2,+1.,theFieldMap);

  // Propagate to the closest approach point, with the constraint that 
  // the particles should pass through the  first hit
  myElecL.propagateToNominalVertex(thePos1);
  myPosiL.propagateToNominalVertex(thePos1);

  theMom2 *= 1000.0;//ptmax
  // The corresponding RawParticles (to be propagated) for e- and e+
  ParticlePropagator myElecH(theMom2,thePos2,-1.,theFieldMap);
  ParticlePropagator myPosiH(theMom2,thePos2,+1.,theFieldMap);

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

  if ( myElecL.R() < originRadius[algo] && 
       fabs(myElecL.Z()) < originHalfLength[algo] ) return true;
  if ( myPosiL.R() < originRadius[algo] && 
       fabs(myPosiL.Z()) < originHalfLength[algo] ) return true;
  if ( myElecH.R() < originRadius[algo] && 
       fabs(myElecH.Z()) < originHalfLength[algo] ) return true;
  if ( myPosiH.R() < originRadius[algo] && 
       fabs(myPosiH.Z()) < originHalfLength[algo] ) return true;

  return false;
}
