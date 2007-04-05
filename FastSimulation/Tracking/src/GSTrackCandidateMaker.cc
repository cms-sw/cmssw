#include <memory>
#include <string>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"
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
#include "RecoTracker/TrackProducer/interface/TrackingRecHitLessFromGlobalPosition.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"
#include "TrackingTools/TrackFitters/interface/RecHitSorter.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "DataFormats/GeometrySurface/interface/MediumProperties.h"
#include "DataFormats/GeometrySurface/interface/BoundSurface.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

//for debug only 
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"
//#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h" 
#include "DataFormats/SiStripDetId/interface/TIBDetId.h" 
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h" 
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
//



using namespace edm;
using namespace std;

#define FAMOS_DEBUG

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

    int nFittableTracks = 0;
    int nCandidates = 0;

    std::auto_ptr<TrackCandidateCollection> output(new TrackCandidateCollection);    

    edm::Handle<SimTrackContainer> theSimTracks;
    e.getByType<SimTrackContainer>(theSimTracks);

    edm::Handle<SimVertexContainer> theSimVtx;
    e.getByType(theSimVtx);
    /*
     e.getByLabel("g4SimHits",simvertices_handle);
     edm::SimVertexContainer const* simvertices = simvertices_handle.product();
    */
#ifdef FAMOS_DEBUG
    std::cout << " Step A: SimTracks found " << theSimTracks->size() << std::endl;
#endif
    
    edm::Handle<SiTrackerGSRecHit2DCollection> theGSRecHits;
    std::string hitProducer = conf_.getParameter<std::string>("HitProducer");
    e.getByLabel(hitProducer, theGSRecHits);
    // skip event if empty RecHit collection
    if(theGSRecHits->size() == 0) return;
#ifdef FAMOS_DEBUG
    std::cout << " Step B: GS RecHits found " << theGSRecHits->size() << std::endl;
#endif
    
    
#ifdef FAMOS_DEBUG
    std::cout << " Step C: Map Rechit by SimTrack ID " << std::endl;
#endif
    // Step C: Fill the vectors of GSRecHits belonging to the same SimTrack
    // map of the vector of GSRecHits belonging to the same SimTrack
    std::map< int, std::vector< SiTrackerGSRecHit2D*>, std::less<int> > mapRecHitsToSimTrack;
    //
    edm::OwnVector<SiTrackerGSRecHit2D>::const_iterator theRecHitIteratorBegin = theGSRecHits->begin();
    edm::OwnVector<SiTrackerGSRecHit2D>::const_iterator theRecHitIteratorEnd   = theGSRecHits->end();

    for(edm::OwnVector<SiTrackerGSRecHit2D>::const_iterator iRecHit = theRecHitIteratorBegin;
	iRecHit != theRecHitIteratorEnd; ++iRecHit) { // loop on GSRecHits
      
      int simTrackId = iRecHit->simtrackId();

      //PAT ADDITION difference of one unit in the ID's (to be checked again)
      //     simTrackId++;
      
      /*
#ifdef FAMOS_DEBUG
      std::cout << "GSRecHit from Sim Track " << simTrackId
		<< " in det " << iRecHit->geographicalId().rawId() << std::endl;
#endif
      */

      mapRecHitsToSimTrack[simTrackId].push_back(const_cast<SiTrackerGSRecHit2D*>(&(*iRecHit)));
    }
    
#ifdef FAMOS_DEBUG
    std::cout << " Step D Loop on SimTrack's to construct candidates" << std::endl;
#endif
    
    for(SimTrackContainer::const_iterator iTrack = theSimTracks->begin(); iTrack != theSimTracks->end(); iTrack++)
      { 
	int iSimTrack = iTrack->trackId();	
	std::map<int, std::vector< SiTrackerGSRecHit2D*> >::const_iterator it = mapRecHitsToSimTrack.find(iSimTrack);
	std::vector< SiTrackerGSRecHit2D*> mappedRecHits;
	mappedRecHits.clear();

	// Create OwnVector with sorted GSRecHit's
	OwnVector<TrackingRecHit> recHits;
	recHits.clear();

	TransientTrackingRecHit::ConstRecHitContainer recHitContainer;
	recHitContainer.clear();

	if(it != mapRecHitsToSimTrack.end()) {
	  mappedRecHits =  it->second;

#ifdef FAMOS_DEBUG
	  std::cout << " SimTrack Id = " << iSimTrack << " contains " << mappedRecHits.size() << " GS RecHits" << std::endl; 
#endif
	  if(mappedRecHits.size()==0) continue;
	  
	  std::vector<SiTrackerGSRecHit2D*>::iterator iHitIter    = mappedRecHits.begin(); 
	  std::vector<SiTrackerGSRecHit2D*>::iterator iHitIterEnd = mappedRecHits.end();
	  
	  for(; iHitIter!=iHitIterEnd; ++iHitIter){
	    const GeomDet* geomDet( theGeometry->idToDet( (*iHitIter)->geographicalId() ) );

	    /*	    
	      #ifdef FAMOS_DEBUG
	      std::cout << " Hit in detector " << geomDet->geographicalId().rawId() << std::endl;
	      #endif
	    */
	    recHitContainer.push_back( GenericTransientTrackingRecHit::build( geomDet , (**iHitIter).clone() ) );
	  }
	}

	// A.S.: do not sort them with this rechit sorter
	//	TransientTrackingRecHit::ConstRecHitContainer sortedRecHits;
	//	sortedRecHits = theRecHitSorter->sortHits(recHitContainer, alongMomentum);
	
// #ifdef FAMOS_DEBUG
// 	std::cout << "Sorted RecHits for SimTrack = " << iSimTrack << " are " << sortedRecHits.size() << std::endl;
// #endif
	
	for(TransientTrackingRecHit::ConstRecHitContainer::iterator iSortedTRecHit = recHitContainer.begin();
	    iSortedTRecHit < recHitContainer.end();
	    iSortedTRecHit++ ) {

#ifdef FAMOS_DEBUG
	  DetId detId =  (*iSortedTRecHit)->geographicalId();
	  unsigned int subdetId = static_cast<unsigned int>(detId.subdetId()); 
	  int layerNumber=0;
	  if ( subdetId == StripSubdetector::TIB) 
	    { 
	      TIBDetId tibid(detId.rawId()); 
	      layerNumber = tibid.layer();
	    }
	  else if ( subdetId ==  StripSubdetector::TOB )
	    { 
	      TOBDetId tobid(detId.rawId()); 
	      layerNumber = tobid.layer();
	    }
	  else if ( subdetId ==  StripSubdetector::TID) 
	    { 
	      TIDDetId tidid(detId.rawId());
	      layerNumber = tidid.wheel();
	    }
	  else if ( subdetId ==  StripSubdetector::TEC )
	    { 
	      TECDetId tecid(detId.rawId()); 
	      layerNumber = tecid.wheel(); 
	    }
	  else if ( subdetId ==  PixelSubdetector::PixelBarrel ) 
	    { 
	      PXBDetId pxbid(detId.rawId()); 
	      layerNumber = pxbid.layer();  
	    }
	  else if ( subdetId ==  PixelSubdetector::PixelEndcap ) 
	    { 
	      PXFDetId pxfid(detId.rawId()); 
	      layerNumber = pxfid.disk();  
	    }
	  
	  std::cout << "Added RecHit from detid " << detId.rawId() << " subdet = " << detId.subdetId() 
		    << " layer = " << layerNumber << std::endl;

#endif

	  TrackingRecHit* aRecHit( (*iSortedTRecHit)->hit()->clone() );
	  recHits.push_back( aRecHit );
	}

	if(recHits.size()<3) {

#ifdef FAMOS_DEBUG
	  std::cout << "******** Not enough hits to fit the track --> Skip Track " << std::endl;
#endif
	  continue;
	}
	nFittableTracks++;
	

	//A.S.: DEBUG: sort hits using the TrackingRecHitLessFromGlobalPosition sorter
	recHits.sort(TrackingRecHitLessFromGlobalPosition(theGeometry.product(),alongMomentum));

	//
	// Create a starting trajectory with SimTrack + first RecHit

	// take the SimTrack parameters
	int vertexIndex = (*iTrack).vertIndex();

#ifdef FAMOS_DEBUG
	std::cout << " SimTrack = " << iSimTrack << "\tVERT ind = " << vertexIndex << 
	  " (x,y,z)= (" <<  (*theSimVtx)[vertexIndex].position() << ")" << std::endl;
#endif

       	GlobalPoint  position((*theSimVtx)[vertexIndex].position().x(),
			      (*theSimVtx)[vertexIndex].position().y(),
			      (*theSimVtx)[vertexIndex].position().z());
	

	GlobalVector momentum( (*iTrack).momentum().x() , (*iTrack).momentum().y() , (*iTrack).momentum().z() );
	float        charge   = (*iTrack).charge();
	GlobalTrajectoryParameters initialParams(position,momentum,(int)charge,&*theMagField);
	AlgebraicSymMatrix errorMatrix(5,1);
	//why?
	errorMatrix = errorMatrix * 10;

#ifdef FAMOS_DEBUG
	std::cout << "GSTrackCandidateMaker: SimTrack parameters " << std::endl;
	std::cout << "\t\t pT  = " << (*iTrack).momentum().perp() << std::endl;
	std::cout << "\t\t eta = " << (*iTrack).momentum().eta()  << std::endl;
	std::cout << "\t\t phi = " << (*iTrack).momentum().phi()  << std::endl;
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
	TrackCandidate newTrackCandidate(recHits, TrajectorySeed(*initialState, recHits, alongMomentum), *initialState );
	// Log

#ifdef FAMOS_DEBUG
	std::cout << "\tSeed Information " << std::endl;
	std::cout << "\tSeed Direction = " << TrajectorySeed(*initialState, recHits, alongMomentum).direction() << std::endl;
	std::cout << "\tSeed StartingDet = " << TrajectorySeed(*initialState, recHits, alongMomentum).startingState().detId() << std::endl;

	std::cout << "\tTrajectory Parameters " << std::endl;
	std::cout << "\t\t detId  = " << newTrackCandidate.trajectoryStateOnDet().detId()                        << std::endl;
	std::cout << "\t\t loc.px = " << newTrackCandidate.trajectoryStateOnDet().parameters().momentum().x()    << std::endl;
	std::cout << "\t\t loc.py = " << newTrackCandidate.trajectoryStateOnDet().parameters().momentum().y()    << std::endl;
	std::cout << "\t\t loc.pz = " << newTrackCandidate.trajectoryStateOnDet().parameters().momentum().z()    << std::endl;
       	std::cout << "\t\t error  = ";
	for(std::vector< float >::const_iterator iElement = newTrackCandidate.trajectoryStateOnDet().errorMatrix().begin();
	    iElement < newTrackCandidate.trajectoryStateOnDet().errorMatrix().end();
	    iElement++) {
	  std::cout << "\t" << *iElement;
	}
	std::cout << std::endl;
#endif
	output->push_back(newTrackCandidate);
	nCandidates++;
      }
    
    
    std::cout << " GSTrackCandidateMaker: \tTotal Fittable =  " << nFittableTracks << "\t Total Candidates = " << nCandidates << std::endl;
    e.put(output);
    
#ifdef FAMOS_DEBUG
    std::cout << " GSTrackCandidateMaker produce end " << std::endl;
#endif
  }
}

