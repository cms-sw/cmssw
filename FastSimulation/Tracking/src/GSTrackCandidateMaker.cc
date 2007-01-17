#include <memory>
#include <string>

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "FastSimulation/Tracking/interface/GSTrackCandidateMaker.h"
#include "RecoTracker/CkfPattern/interface/TransientInitialStateEstimator.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/TrackFitters/interface/RecHitSorter.h"
#include "Geometry/Surface/interface/BoundSurface.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "Geometry/Surface/interface/MediumProperties.h"

using namespace edm;
using namespace std;

//#define FAMOS_DEBUG

namespace cms{
  GSTrackCandidateMaker::GSTrackCandidateMaker(edm::ParameterSet const& conf) : 

    conf_(conf)
  {  
#ifdef FAMOS_DEBUG
    std::cout << "GSTrackCandidateMaker created" << std::endl;
#endif
    produces<TrackCandidateCollection>();
    theRecHitSorter = new RecHitSorter();
  }

  
  // Virtual destructor needed.
  GSTrackCandidateMaker::~GSTrackCandidateMaker() {
    // do nothing
#ifdef FAMOS_DEBUG
    std::cout << "GSTrackCandidateMaker destructed" << std::endl;
#endif
  }  

  void GSTrackCandidateMaker::beginJob (EventSetup const & es)
  {
    //services
    es.get<TrackerRecoGeometryRecord>().get(theGeomSearchTracker);
    es.get<IdealMagneticFieldRecord>().get(theMagField);
    es.get<TrackerDigiGeometryRecord>().get(theGeometry);
  }
  
  // Functions that gets called by framework every event
  void GSTrackCandidateMaker::produce(edm::Event& e, const edm::EventSetup& es)
  {        
#ifdef FAMOS_DEBUG
    std::cout << "################################################################" << std::endl;
    std::cout << " GSTrackCandidateMaker produce init " << std::endl;
#endif
    edm::Handle<SimTrackContainer> theSimTracks;
    e.getByType<SimTrackContainer>(theSimTracks);
#ifdef FAMOS_DEBUG
    std::cout << " SimTracks found " << theSimTracks->size() << std::endl;
#endif
    
#ifdef FAMOS_DEBUG
    std::cout << " Step A " << std::endl;
#endif
    // Step A: Retrieve GSRecHits
    edm::Handle<SiTrackerGSRecHit2DCollection> theGSRecHits;
    std::string hitProducer = conf_.getParameter<std::string>("HitProducer");
    e.getByLabel(hitProducer, theGSRecHits);
    // skip event if empty RecHit collection
    if(theGSRecHits->size() == 0) return;
    
#ifdef FAMOS_DEBUG
    std::cout << " Step B " << std::endl;
#endif
    // Step B: Create empty output collection
    std::auto_ptr<TrackCandidateCollection> output(new TrackCandidateCollection);    
    
#ifdef FAMOS_DEBUG
    std::cout << " Step C " << std::endl;
#endif
    // Step C: Fill the vectors of GSRecHits belonging to the same SimTrack
    // map of the vector of GSRecHits belonging to the same SimTrack
    std::map< int, std::vector< SiTrackerGSRecHit2D*>, std::less<int> > mapRecHitsToSimTrack;
    //
    edm::OwnVector<SiTrackerGSRecHit2D>::const_iterator theRecHitIteratorBegin = theGSRecHits->begin();
    edm::OwnVector<SiTrackerGSRecHit2D>::const_iterator theRecHitIteratorEnd   = theGSRecHits->end();
    int nSimTracks = 0;
    for(edm::OwnVector<SiTrackerGSRecHit2D>::const_iterator iRecHit = theRecHitIteratorBegin;
	iRecHit != theRecHitIteratorEnd;
	++iRecHit) { // loop on GSRecHits
      int simTrackId = iRecHit->simtrackId();
#ifdef FAMOS_DEBUG
      std::cout << "GSRecHit from Sim Track " << simTrackId
		<< " in det " << iRecHit->geographicalId().rawId() << std::endl;
#endif
      mapRecHitsToSimTrack[simTrackId].push_back(const_cast<SiTrackerGSRecHit2D*>(&(*iRecHit)));
      if(simTrackId > nSimTracks) nSimTracks = simTrackId;
    }
#ifdef FAMOS_DEBUG
    std::cout << "GSRecHits from " << nSimTracks << " SimTracks rearranged" << std::endl;
#endif
    //
    
#ifdef FAMOS_DEBUG
    std::cout << " Step D " << std::endl;
#endif
    // Step D: Loop on SimTrack's to construct candidates
    //
    int iSimTrack = 0;
    for(SimTrackContainer::const_iterator iTrack = theSimTracks->begin(); iTrack != theSimTracks->end(); iTrack++){ // loop on SimTracks
      iSimTrack++;
      // Create OwnVector with sorted GSRecHit's
      OwnVector<TrackingRecHit> recHits;
      recHits.clear();
      int iHit = -1;
      TransientTrackingRecHit::ConstRecHitContainer recHitContainer;
      recHitContainer.clear();
      for(std::vector<SiTrackerGSRecHit2D*>::iterator iRecHit = mapRecHitsToSimTrack[iSimTrack].begin();
	  iRecHit < mapRecHitsToSimTrack[iSimTrack].end();
	  iRecHit++) {
	iHit++;
	const GeomDet* geomDet( theGeometry->idToDet( (*iRecHit)->geographicalId() ) );
#ifdef FAMOS_DEBUG
	std::cout << " Hit in detector " << geomDet->geographicalId().rawId() << std::endl;
#endif
	recHitContainer.push_back( GenericTransientTrackingRecHit::build( geomDet , (**iRecHit).clone() ) );
      }
#ifdef FAMOS_DEBUG
      std::cout << "GSRecHits of SimTrack " << iSimTrack << " found " << iHit+1 << " before sorting RecHits are " << recHitContainer.size() << std::endl;
#endif
      TransientTrackingRecHit::ConstRecHitContainer sortedRecHits = theRecHitSorter->sortHits(recHitContainer, alongMomentum);
#ifdef FAMOS_DEBUG
      std::cout << "Sorted RecHits are " << sortedRecHits.size() << std::endl;
#endif
      //
      for(TransientTrackingRecHit::ConstRecHitContainer::iterator iSortedTRecHit = sortedRecHits.begin();
	  iSortedTRecHit < sortedRecHits.end();
	  iSortedTRecHit++ ) {
#ifdef FAMOS_DEBUG
	std::cout << "Added RecHit from detector " << (*iSortedTRecHit)->geographicalId().rawId() << std::endl;
#endif
	TrackingRecHit* aRecHit( (*iSortedTRecHit)->hit()->clone() );
	recHits.push_back( aRecHit );
      }
#ifdef FAMOS_DEBUG
      std::cout << " RecHits: sorted total " << recHits.size() << std::endl;
#endif
      if(recHits.size()<3) {
#ifdef FAMOS_DEBUG
	std::cout << "******** Not enough hits to fit the track --> Skip Track " << std::endl;
#endif
	break;
      }
      //
      // Create a starting trajectory with SimTrack + first RecHit
#ifdef FAMOS_DEBUG
      std::cout << "GSTrackCandidateMaker: GeomDetUnit " << std::endl;
#endif
      const GeomDetUnit* initialLayer = theGeometry->idToDetUnit( recHits.front().geographicalId() );
      // take the SimTrack parameters
#ifdef FAMOS_DEBUG
      std::cout << "GSTrackCandidateMaker: SimTrack parameters " << std::endl;
      std::cout << "\t\t pT  = " << (*iTrack).momentum().perp() << std::endl;
      std::cout << "\t\t eta = " << (*iTrack).momentum().eta()  << std::endl;
      std::cout << "\t\t phi = " << (*iTrack).momentum().phi()  << std::endl;
#endif
      GlobalPoint  position(0.,0.,0.);
      //      int vertexIndex = (*iTrack).vertIndex();
      GlobalVector momentum( (*iTrack).momentum().x() , (*iTrack).momentum().y() , (*iTrack).momentum().z() );
      float        charge   = (*iTrack).charge();
      //
      GlobalTrajectoryParameters initialParams(position,momentum,(int)charge,&*theMagField);
      AlgebraicSymMatrix errorMatrix(5,1);
      errorMatrix = errorMatrix * 10;
#ifdef FAMOS_DEBUG
      std::cout << "GSTrackCandidateMaker: AlgebraicSymMatrix " << errorMatrix << std::endl;
#endif
      CurvilinearTrajectoryError initialError(errorMatrix);
      // Construct TSOS from FTS + Surface
      FreeTrajectoryState initialFTS(initialParams, initialError);
      //      BoundSurface initialSurface = initialLayer->surface();
#ifdef FAMOS_DEBUG
      std::cout << "GSTrackCandidateMaker: FTS momentum " << initialFTS.momentum() << std::endl;
#endif
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
      std::cout << "GSTrackCandidateMaker: TSOS X0[cm] = "           << initialTSOS.surface().mediumProperties()->radLen() << std::endl;
#endif
      //
      TrajectoryStateTransform transformer;
      PTrajectoryStateOnDet* initialState( transformer.persistentState( initialTSOS, recHits.front().geographicalId().rawId() ) );
#ifdef FAMOS_DEBUG
      std::cout << "GSTrackCandidateMaker: detid " << recHits.front().geographicalId().rawId() << std::endl;
      std::cout << "GSTrackCandidateMaker: PTSOS detId " << initialState->detId() << std::endl;
      std::cout << "GSTrackCandidateMaker: PTSOS local momentum " << initialState->parameters().momentum() << std::endl;
#endif
      //
      // Track Candidate stored
      TrackCandidate newTrackCandidate(recHits, TrajectorySeed(), *initialState );
      // Log
#ifdef FAMOS_DEBUG
      std::cout << "\tTrajectory Parameters " << std::endl;
      std::cout << "\t\t detId  = " << newTrackCandidate.trajectoryStateOnDet().detId()                        << std::endl;
      std::cout << "\t\t loc.px = " << newTrackCandidate.trajectoryStateOnDet().parameters().momentum().x()    << std::endl;
      std::cout << "\t\t loc.py = " << newTrackCandidate.trajectoryStateOnDet().parameters().momentum().y()    << std::endl;
      std::cout << "\t\t loc.pz = " << newTrackCandidate.trajectoryStateOnDet().parameters().momentum().z()    << std::endl;
      std::cout << "\t\t error  = ";
#endif

#ifdef FAMOS_DEBUG
      for(std::vector< float >::const_iterator iElement = newTrackCandidate.trajectoryStateOnDet().errorMatrix().begin();
	  iElement < newTrackCandidate.trajectoryStateOnDet().errorMatrix().end();
	  iElement++) {
	std::cout << "\t" << *iElement;
      }
#endif

#ifdef FAMOS_DEBUG
      std::cout << std::endl;
#endif
      //
      output->push_back(newTrackCandidate);
    }
    
    std::cout << " Step E " << std::endl;
    // Step E: write output to file
    e.put(output);
    
#ifdef FAMOS_DEBUG
    std::cout << " GSTrackCandidateMaker produce end " << std::endl;
#endif
  }
}

