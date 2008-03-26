/** \class MuonDetLayerMeasurements
 *  The class to access recHits and TrajectoryMeasurements from DetLayer.
 *
 *  $Date: 2007/11/20 19:05:32 $
 *  $Revision: 1.22 $
 *  \author C. Liu, R. Bellan, N. Amapane
 *
 */

#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"

//#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h" 
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


//typedef TransientTrackingRecHit::RecHitPointer RecHitPointer;
//typedef TransientTrackingRecHit::RecHitContainer RecHitContainer;
typedef MuonTransientTrackingRecHit::MuonRecHitPointer MuonRecHitPointer;
typedef MuonTransientTrackingRecHit::MuonRecHitContainer MuonRecHitContainer;



MuonDetLayerMeasurements::~MuonDetLayerMeasurements() {}


MuonDetLayerMeasurements::MuonDetLayerMeasurements(edm::InputTag dtlabel, 
						   edm::InputTag csclabel, 
						   edm::InputTag rpclabel,
						   bool enableDT, bool enableCSC, bool enableRPC): 
  theDTRecHitLabel(dtlabel),
  theCSCRecHitLabel(csclabel),
  theRPCRecHitLabel(rpclabel),
  enableDTMeasurement(enableDT),
  enableCSCMeasurement(enableCSC),
  enableRPCMeasurement(enableRPC),
  theEvent(0){}


MuonTransientTrackingRecHit::MuonRecHitContainer MuonDetLayerMeasurements::recHits(const GeomDet* geomDet, const edm::Event& iEvent) const {

  MuonRecHitContainer muonRecHits;
  
  DetId geoId = geomDet->geographicalId();
  
  if (geoId.subdetId()  == MuonSubdetId::DT ) {
    if(!enableDTMeasurement) return muonRecHits;

    // Get the DT-Segment collection from the Event
    edm::Handle<DTRecSegment4DCollection> dtRecHits;
    iEvent.getByLabel(theDTRecHitLabel, dtRecHits);  
    
    // Create the ChamberId
    DTChamberId chamberId(geoId.rawId());
    LogTrace("Muon|RecoMuon|MuonDetLayerMeasurements") << "(DT): "<<chamberId<<std::endl;
    
    // Get the DT-Segment which relies on this chamber
    DTRecSegment4DCollection::range range = dtRecHits->get(chamberId);
    
    // Create the MuonTransientTrackingRechit
    for (DTRecSegment4DCollection::const_iterator rechit = range.first; rechit!=range.second;++rechit){
      //MuonRecHitPointer muonRecHit = MuonTransientTrackingRecHit::specificBuild(geomDet, (&(*rechit)));
      muonRecHits.push_back(  MuonTransientTrackingRecHit::specificBuild(geomDet, (&(*rechit))) );
    }
  }
  
  else if (geoId.subdetId()  == MuonSubdetId::CSC) {
    if(!enableCSCMeasurement) return muonRecHits;

    // Get the CSC-Segment collection from the event
    edm::Handle<CSCSegmentCollection> cscSegments;
    iEvent.getByLabel(theCSCRecHitLabel, cscSegments); 

    // Create the chamber Id
    CSCDetId chamberId(geoId.rawId());
    LogTrace("Muon|RecoMuon|MuonDetLayerMeasurements") << "(CSC): "<<chamberId<<std::endl;

    // Get the CSC-Segment which relies on this chamber
    CSCSegmentCollection::range range = cscSegments->get(chamberId);
    
    // Create the MuonTransientTrackingRecHit
    for (CSCSegmentCollection::const_iterator rechit = range.first; rechit!=range.second; ++rechit){
      muonRecHits.push_back(  MuonTransientTrackingRecHit::specificBuild(geomDet, (&(*rechit))) ); 
    }    
  }
  
  else if (geoId.subdetId()  == MuonSubdetId::RPC ) {
    if(!enableRPCMeasurement) return muonRecHits;
    
    // Get the CSC-Segment collection from the event
    edm::Handle<RPCRecHitCollection> rpcRecHits;
    iEvent.getByLabel(theRPCRecHitLabel, rpcRecHits); 
    
    // Create the chamber Id
    RPCDetId chamberId(geoId.rawId());
    LogTrace("Muon|RecoMuon|MuonDetLayerMeasurements") << "(RPC): "<<chamberId<<std::endl;
    
    // Get the RPC-Segment which relies on this chamber
    RPCRecHitCollection::range range = rpcRecHits->get(chamberId);
    
    // Create the MuonTransientTrackingRecHit
    for (RPCRecHitCollection::const_iterator rechit = range.first; rechit!=range.second; ++rechit){
      muonRecHits.push_back(  MuonTransientTrackingRecHit::specificBuild(geomDet, (&(*rechit))) );
    }
  }
  else {
    // wrong type
    edm::LogWarning("MuonDetLayerMeasurements")<<"The DetLayer is not a valid Muon DetLayer. ";
  }
  
  return muonRecHits;
}


MeasurementContainer
MuonDetLayerMeasurements::measurements( const DetLayer * layer,
                                        const GeomDet * det,
                                        const TrajectoryStateOnSurface& stateOnDet,
                                        const MeasurementEstimator& est,
                                        const edm::Event& iEvent) const 
{
  MeasurementContainer result;
  MuonRecHitContainer rhs = recHits(det, iEvent);
  for (MuonRecHitContainer::const_iterator irh = rhs.begin(); irh!=rhs.end(); irh++) {
    MeasurementEstimator::HitReturnType estimate = est.estimate(stateOnDet, (**irh));
    //if (estimate.first)
    //{
      result.push_back(TrajectoryMeasurement(stateOnDet,(*irh).get(),
                                             estimate.second,layer));
    //}
  }

  if (!result.empty()) {
    sort( result.begin(), result.end(), TrajMeasLessEstim());
  }

  return result;
}



MeasurementContainer
MuonDetLayerMeasurements::measurements( const DetLayer* layer,
					const TrajectoryStateOnSurface& startingState,
					const Propagator& prop,
					const MeasurementEstimator& est,
					const edm::Event& iEvent) const {
  
  MeasurementContainer result;
  
  std::vector<DetWithState> dss = layer->compatibleDets(startingState, prop, est);
  LogTrace("RecoMuon")<<"compatibleDets: "<<dss.size()<<std::endl;
  
  std::vector<DetWithState>::const_iterator detWithStateItr = dss.begin(),
                                            detWithStateEnd = dss.end();
  for( ; detWithStateItr != detWithStateEnd; ++detWithStateItr)
  {
    MeasurementContainer detMeasurements 
      = measurements(layer, detWithStateItr->first, 
                     detWithStateItr->second, est, iEvent);
    result.insert(result.end(), detMeasurements.begin(), detMeasurements.end());
  }
  if (!result.empty()) {
    sort( result.begin(), result.end(), TrajMeasLessEstim());
  }
  return result;
}


MeasurementContainer
MuonDetLayerMeasurements::fastMeasurements( const DetLayer* layer,
					    const TrajectoryStateOnSurface& theStateOnDet,
					    const TrajectoryStateOnSurface& startingState,
					    const Propagator& prop,
					    const MeasurementEstimator& est,
					    const edm::Event& iEvent) const {
  MeasurementContainer result;
  MuonRecHitContainer rhs = recHits(layer, iEvent);
  for (MuonRecHitContainer::const_iterator irh = rhs.begin(); irh!=rhs.end(); irh++) {
    MeasurementEstimator::HitReturnType estimate = est.estimate(theStateOnDet, (**irh));
    if (estimate.first)
    {
      result.push_back(TrajectoryMeasurement(theStateOnDet,(*irh).get(),
                                             estimate.second,layer));
    }
  }

  if (!result.empty()) {
    sort( result.begin(), result.end(), TrajMeasLessEstim());
  }

  return result;
}

///measurements method if already got the Event 
MeasurementContainer
MuonDetLayerMeasurements::measurements( const DetLayer* layer,
					const TrajectoryStateOnSurface& startingState,
					const Propagator& prop,
					const MeasurementEstimator& est) const {
  checkEvent();
  return measurements(layer, startingState, prop, est, *theEvent);
}

///fastMeasurements method if already got the Event
MeasurementContainer
MuonDetLayerMeasurements::fastMeasurements( const DetLayer* layer,
					    const TrajectoryStateOnSurface& theStateOnDet,
					    const TrajectoryStateOnSurface& startingState,
					    const Propagator& prop,
					    const MeasurementEstimator& est) const {
  checkEvent();
  return fastMeasurements(layer, theStateOnDet, startingState, prop, est, *theEvent); 
}


std::vector<TrajectoryMeasurementGroup>
MuonDetLayerMeasurements::groupedMeasurements( const DetLayer* layer,
                  const TrajectoryStateOnSurface& startingState,
                  const Propagator& prop,
                  const MeasurementEstimator& est,
                  const edm::Event& iEvent) const
{
  std::vector<TrajectoryMeasurementGroup> result;
  // if we want to use the concept of InvalidRecHits,
  // we can reuse LayerMeasurements from TrackingTools/MeasurementDet
  std::vector<DetGroup> groups( layer->groupedCompatibleDets( startingState, prop, est));
  for (std::vector<DetGroup>::const_iterator grp=groups.begin(); grp!=groups.end(); grp++) {

    std::vector<TrajectoryMeasurement> groupMeasurements;
    for (DetGroup::const_iterator detAndStateItr=grp->begin();
         detAndStateItr !=grp->end(); detAndStateItr++) 
    {
      std::vector<TrajectoryMeasurement> detMeasurements 
        = measurements(layer, detAndStateItr->det(), detAndStateItr->trajectoryState(), est, iEvent);
      groupMeasurements.insert(groupMeasurements.end(), detMeasurements.begin(), detMeasurements.end());
    }
    if (!groupMeasurements.empty()) {
      std::sort( groupMeasurements.begin(), groupMeasurements.end(), TrajMeasLessEstim());
    }

    result.push_back(TrajectoryMeasurementGroup(groupMeasurements, *grp));
  }

  return result;
}


std::vector<TrajectoryMeasurementGroup>
MuonDetLayerMeasurements::groupedMeasurements( const DetLayer* layer,
                  const TrajectoryStateOnSurface& startingState,
                  const Propagator& prop,
                  const MeasurementEstimator& est) const
{
  checkEvent();
  return groupedMeasurements(layer, startingState, prop,  est, *theEvent);
}


///set event
void MuonDetLayerMeasurements::setEvent(const edm::Event& event) {
  theEvent = &event;
}


void MuonDetLayerMeasurements::checkEvent() const
{
  if(!theEvent)
  {
    throw cms::Exception("MuonDetLayerMeasurements") << "The event has not been set";
  }
}


MuonRecHitContainer MuonDetLayerMeasurements::recHits(const DetLayer* layer, const edm::Event& iEvent) const
{
  MuonRecHitContainer rhs;
  
  std::vector <const GeomDet*> gds = layer->basicComponents();

  for (std::vector<const GeomDet*>::const_iterator igd = gds.begin(); 
       igd != gds.end(); igd++) 
  {
    MuonRecHitContainer detHits = recHits(*igd, iEvent);
    rhs.insert(rhs.end(), detHits.begin(), detHits.end());
  }

  return rhs;
}

MuonRecHitContainer MuonDetLayerMeasurements::recHits(const DetLayer* layer) const
{
  MuonRecHitContainer result;
  if (theEvent) return recHits(layer, *theEvent);
  else return result;
}

