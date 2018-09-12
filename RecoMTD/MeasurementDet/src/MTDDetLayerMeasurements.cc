/** \class MTDDetLayerMeasurements
 *  The class to access recHits and TrajectoryMeasurements from DetLayer.
 *
 *  \author B. Tannenwald 
 *  Adapted from RecoMuon version.
 *
 */

#include "RecoMTD/MeasurementDet/interface/MTDDetLayerMeasurements.h"

#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h" 
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"



//typedef TransientTrackingRecHit::MTDRecHitPointer MTDRecHitPointer;
typedef std::shared_ptr<GenericTransientTrackingRecHit> MTDRecHitPointer;
typedef std::vector<GenericTransientTrackingRecHit::RecHitPointer> MTDRecHitContainer;
//typedef TransientTrackingRecHit::MTDRecHitContainer MTDRecHitContainer;



MTDDetLayerMeasurements::MTDDetLayerMeasurements(edm::InputTag mtdlabel,
						 edm::ConsumesCollector& iC): 
  theMTDRecHits(),
  theMTDEventCacheID(0),
  theEvent(nullptr)
{

  mtdToken_ = iC.consumes<MTDTrackingRecHit>(mtdlabel);
  
  static std::atomic<int> procInstance{0};
  std::ostringstream sMTD;
  sMTD<<"MTDDetLayerMeasurements::checkMTDRecHits::" << procInstance;
  procInstance++;
}

MTDDetLayerMeasurements::~MTDDetLayerMeasurements(){}

MTDRecHitContainer MTDDetLayerMeasurements::recHits(const GeomDet* geomDet, 
			                              const edm::Event& iEvent)
{
  DetId geoId = geomDet->geographicalId();
  theEvent = &iEvent;
  MTDRecHitContainer result;
  //GenericTransientTrackingRecHit result;

  checkMTDRecHits();
  
  // Create the ChamberId
  DetId detId(geoId.rawId());
  LogDebug("Track|RecoMTD|MTDDetLayerMeasurements") << "(MTD): "<<detId<<std::endl;
  
  // Get the MTD-Segment which relies on this chamber
  //auto cmp = [](const unsigned one, const unsigned two) -> bool { return one < two; };
  auto detset = (*theMTDRecHits)[detId];
  
  for (auto rechit = detset.begin(); rechit!=detset.end();++rechit)
    //result.push_back(TransientTrackingRecHit::specificBuild(geomDet,&*rechit));
    result.push_back(GenericTransientTrackingRecHit::build(geomDet, &*rechit));
    //result.push_back(TrackingRecHit::RecHitPointer::build(geomDet, &*rechit));

  /*
  if (geoId.subdetId()  == MuonSubdetId::DT) {
    if(enableDTMeasurement) 
    {
      checkDTRecHits();
    
      // Create the ChamberId
      DTChamberId detId(geoId.rawId());
      LogDebug("Muon|RecoMuon|MTDDetLayerMeasurements") << "(DT): "<<detId<<std::endl;
    
      // Get the DT-Segment which relies on this chamber
      DTRecSegment4DCollection::range range = theDTRecHits->get(detId);
    
      // Create the MuonTransientTrackingRechit
      for (DTRecSegment4DCollection::const_iterator rechit = range.first; 
           rechit!=range.second;++rechit)
        result.push_back(MuonTransientTrackingRecHit::specificBuild(geomDet,&*rechit));
    }
  }
  */
  /*
  else if (geoId.subdetId()  == MuonSubdetId::CSC) {
    if(enableCSCMeasurement)
    {
      checkCSCRecHits();

      // Create the chamber Id
      CSCDetId detId(geoId.rawId());
      LogDebug("Track|RecoMTD|MTDDetLayerMeasurements") << "(CSC): "<<detId<<std::endl;

      // Get the CSC-Segment which relies on this chamber
      CSCSegmentCollection::range range = theCSCRecHits->get(detId);
    
      // Create the MuonTransientTrackingRecHit
      for (CSCSegmentCollection::const_iterator rechit = range.first; 
           rechit!=range.second; ++rechit)
        result.push_back(MuonTransientTrackingRecHit::specificBuild(geomDet,&*rechit)); 
    }
  }
  */
  /*
  else if (geoId.subdetId()  == MuonSubdetId::RPC) {
    if(enableRPCMeasurement)
    {
      checkRPCRecHits(); 

      // Create the chamber Id
      RPCDetId detId(geoId.rawId());
      LogDebug("Track|RecoMTD|MTDDetLayerMeasurements") << "(RPC): "<<detId<<std::endl;
    
      // Get the RPC-Segment which relies on this chamber
      RPCRecHitCollection::range range = theRPCRecHits->get(detId);
    
      // Create the MuonTransientTrackingRecHit
      for (RPCRecHitCollection::const_iterator rechit = range.first; 
           rechit!=range.second; ++rechit)
        result.push_back(MuonTransientTrackingRecHit::specificBuild(geomDet,&*rechit));
    }
  }
  */
  /*
  else if (geoId.subdetId()  == MuonSubdetId::GEM) {
    if(enableGEMMeasurement)
      {
	checkGEMRecHits(); 

	// Create the chamber Id
	GEMDetId detId(geoId.rawId());

	LogDebug("Track|RecoMTD|MTDDetLayerMeasurements") << "(GEM): "<<detId<<std::endl;

	// Get the GEM-Segment which relies on this chamber
	GEMRecHitCollection::range range = theGEMRecHits->get(detId);

	LogDebug("Track|RecoMTD|MTDDetLayerMeasurements") << "Number of GEM rechits available =  " << theGEMRecHits->size()
							   <<", from chamber: "<< detId<<std::endl;

	// Create the MuonTransientTrackingRecHit
	for (GEMRecHitCollection::const_iterator rechit = range.first; 
	     rechit!=range.second; ++rechit)
	  result.push_back(MuonTransientTrackingRecHit::specificBuild(geomDet,&*rechit));
	LogDebug("Track|RecoMTD|MTDDetLayerMeasurements") << "Number of GEM rechits = " << result.size()<<std::endl;
      }
  }
  */
  /*
  else if (geoId.subdetId()  == MuonSubdetId::ME0) {
    LogDebug("Track|RecoMTD|MTDDetLayerMeasurements") << "(ME0): identified"<<std::endl;
    if(enableME0Measurement)
      {
	checkME0RecHits(); 

	// Create the chamber Id
	ME0DetId detId(geoId.rawId());
    
	// Get the ME0-Segment which relies on this chamber
	// Getting rechits right now, not segments - maybe it should be segments?
	ME0SegmentCollection::range range = theME0RecHits->get(detId);

	LogDebug("Track|RecoMTD|MTDDetLayerMeasurements") << "Number of ME0 rechits available =  " << theME0RecHits->size()
							   <<", from chamber: "<< detId<<std::endl;

	// Create the MuonTransientTrackingRecHit
	for (ME0SegmentCollection::const_iterator rechit = range.first; 
	     rechit!=range.second; ++rechit){
	  LogDebug("Track|RecoMTD|MTDDetLayerMeasurements") << "On ME0 iteration " <<std::endl;
	  result.push_back(MuonTransientTrackingRecHit::specificBuild(geomDet,&*rechit));
	}
	LogDebug("Track|RecoMTD|MTDDetLayerMeasurements") << "Number of ME0 rechits = " << result.size()<<std::endl;
      }
  }
*/
/*
// FIXME, these might be needed for debugging
  else {
    // wrong type
    throw cms::Exception("MTDDetLayerMeasurements") << "The DetLayer with det " << geoId.det() << " subdet " << geoId.subdetId() << " is not a valid MTD DetLayer. ";
    LogDebug("Track|RecoMTD|MTDDetLayerMeasurements")<< "The DetLayer with det " << geoId.det() << " subdet " << geoId.subdetId() << " is not a valid MTD DetLayer. ";
  }
*/
  /*if (enableME0Measurement){
    LogDebug("Track|RecoMTD|MTDDetLayerMeasurements") << "(ME0): enabled"<<std::endl;
  }

  if (enableGEMMeasurement){
    LogDebug("Track|RecoMTD|MTDDetLayerMeasurements") << "(GEM): enabled"<<std::endl;
    }*/
  return result;
}

void MTDDetLayerMeasurements::checkMTDRecHits()
{
  LogDebug("Track|RecoMTD|MTDDetLayerMeasurements") << "Checking MTD RecHits";
  checkEvent();
  auto cacheID = theEvent->cacheIdentifier();
  if (cacheID == theMTDEventCacheID) return;

  {
    theEvent->getByToken(mtdToken_, theMTDRecHits);
    theMTDEventCacheID = cacheID;
  }
  if(!theMTDRecHits.isValid())
  {
    throw cms::Exception("MTDDetLayerMeasurements") << "Cannot get MTD RecHits";
    LogDebug("Track|RecoMTD|MTDDetLayerMeasurements") << "Cannot get MTD RecHits";
  }
}

///measurements method if already got the Event 
MeasurementContainer
MTDDetLayerMeasurements::measurements( const DetLayer* layer,
				       const TrajectoryStateOnSurface& startingState,
				       const Propagator& prop,
				       const MeasurementEstimator& est) {
  checkEvent();
  return measurements(layer, startingState, prop, est, *theEvent);
}


MeasurementContainer
MTDDetLayerMeasurements::measurements( const DetLayer* layer,
				       const TrajectoryStateOnSurface& startingState,
				       const Propagator& prop,
				       const MeasurementEstimator& est,
				       const edm::Event& iEvent) {
  
  MeasurementContainer result;  

  std::vector<DetWithState> dss = layer->compatibleDets(startingState, prop, est);
  LogDebug("RecoMTD")<<"compatibleDets: "<<dss.size()<<std::endl;
  
  for(std::vector<DetWithState>::const_iterator detWithStateItr = dss.begin();
      detWithStateItr != dss.end(); ++detWithStateItr){

    MeasurementContainer detMeasurements 
      = measurements(layer, detWithStateItr->first, 
                     detWithStateItr->second, est, iEvent);
    result.insert(result.end(), detMeasurements.begin(), detMeasurements.end());
  }
  
  if (!result.empty()) sort( result.begin(), result.end(), TrajMeasLessEstim());
  
  return result;
}


MeasurementContainer
MTDDetLayerMeasurements::measurements( const DetLayer* layer,
				       const GeomDet* det,
				       const TrajectoryStateOnSurface& stateOnDet,
				       const MeasurementEstimator& est,
				       const edm::Event& iEvent) {
  MeasurementContainer result;
    
  // Get the Segments which relies on the GeomDet given by compatibleDets
  MTDRecHitContainer mtdRecHits = recHits(det, iEvent);
  
  // Create the Trajectory Measurement
  for(auto rechit = mtdRecHits.begin();
      rechit != mtdRecHits.end(); ++rechit) {
    
    MeasurementEstimator::HitReturnType estimate = est.estimate(stateOnDet,**rechit);
    LogDebug("RecoMTD")<<"Dimension: "<<(*rechit)->dimension()
		       <<" Chi2: "<<estimate.second<<std::endl;
    if (estimate.first) {
      result.push_back(TrajectoryMeasurement(stateOnDet, *rechit,
					     estimate.second,layer));
    }
  }

  if (!result.empty()) sort( result.begin(), result.end(), TrajMeasLessEstim());
   
  return result;
}



MeasurementContainer
MTDDetLayerMeasurements::fastMeasurements( const DetLayer* layer,
					   const TrajectoryStateOnSurface& theStateOnDet,
					   const TrajectoryStateOnSurface& startingState,
					   const Propagator& prop,
					   const MeasurementEstimator& est,
					   const edm::Event& iEvent) {
  MeasurementContainer result;
  MTDRecHitContainer rhs = recHits(layer, iEvent);
  for (auto irh = rhs.begin(); irh!=rhs.end(); irh++) {
    MeasurementEstimator::HitReturnType estimate = est.estimate(theStateOnDet, (**irh));
    if (estimate.first)
      {
	result.push_back(TrajectoryMeasurement(theStateOnDet,(*irh),
					       estimate.second,layer));
      }
  }
  
  if (!result.empty()) {
    sort( result.begin(), result.end(), TrajMeasLessEstim());
  }
  
  return result;
}

///fastMeasurements method if already got the Event
MeasurementContainer
MTDDetLayerMeasurements::fastMeasurements( const DetLayer* layer,
					   const TrajectoryStateOnSurface& theStateOnDet,
					   const TrajectoryStateOnSurface& startingState,
					   const Propagator& prop,
					   const MeasurementEstimator& est) {
  checkEvent();
  return fastMeasurements(layer, theStateOnDet, startingState, prop, est, *theEvent); 
}


std::vector<TrajectoryMeasurementGroup>
MTDDetLayerMeasurements::groupedMeasurements( const DetLayer* layer,
					      const TrajectoryStateOnSurface& startingState,
					      const Propagator& prop,
					      const MeasurementEstimator& est) {
  checkEvent();
  return groupedMeasurements(layer, startingState, prop,  est, *theEvent);
}


std::vector<TrajectoryMeasurementGroup>
MTDDetLayerMeasurements::groupedMeasurements(const DetLayer* layer,
					      const TrajectoryStateOnSurface& startingState,
					      const Propagator& prop,
					      const MeasurementEstimator& est,
					      const edm::Event& iEvent) {
  
  std::vector<TrajectoryMeasurementGroup> result;
  // if we want to use the concept of InvalidRecHits,
  // we can reuse LayerMeasurements from TrackingTools/MeasurementDet
  std::vector<DetGroup> groups(layer->groupedCompatibleDets(startingState, prop, est));

  // this should be fixed either in RecoMuon/MeasurementDet/MTDDetLayerMeasurements or
  // RecoMuon/DetLayers/MuRingForwardDoubleLayer
  // and removed the reverse operation in StandAloneMuonFilter::findBestMeasurements

  for (std::vector<DetGroup>::const_iterator grp=groups.begin(); grp!=groups.end(); ++grp) {
    
    std::vector<TrajectoryMeasurement> groupMeasurements;
    for (DetGroup::const_iterator detAndStateItr=grp->begin();
         detAndStateItr !=grp->end(); ++detAndStateItr) {

      std::vector<TrajectoryMeasurement> detMeasurements 
        = measurements(layer, detAndStateItr->det(), detAndStateItr->trajectoryState(), est, iEvent);
      groupMeasurements.insert(groupMeasurements.end(), detMeasurements.begin(), detMeasurements.end());
    }
    
    if (!groupMeasurements.empty()) 
      std::sort( groupMeasurements.begin(), groupMeasurements.end(), TrajMeasLessEstim());  
    
    result.push_back(TrajectoryMeasurementGroup(groupMeasurements, *grp));
  }

  return result;
}

///set event
void MTDDetLayerMeasurements::setEvent(const edm::Event& event) {
  theEvent = &event;
}


void MTDDetLayerMeasurements::checkEvent() const {
  if(!theEvent)
    throw cms::Exception("MTDDetLayerMeasurements") << "The event has not been set";
}

MTDRecHitContainer MTDDetLayerMeasurements::recHits(const DetLayer* layer, 
						      const edm::Event& iEvent) {
  MTDRecHitContainer rhs;
  
  std::vector <const GeomDet*> gds = layer->basicComponents();

  for (std::vector<const GeomDet*>::const_iterator igd = gds.begin(); 
       igd != gds.end(); igd++) {
    MTDRecHitContainer detHits = recHits(*igd, iEvent);
    rhs.insert(rhs.end(), detHits.begin(), detHits.end());
  }
  return rhs;
}

MTDRecHitContainer MTDDetLayerMeasurements::recHits(const DetLayer* layer) 
{
  checkEvent();
  return recHits(layer, *theEvent);
}

