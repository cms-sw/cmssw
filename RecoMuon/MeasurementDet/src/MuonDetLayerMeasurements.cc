/** \class MuonDetLayerMeasurements
 *  The class to access recHits and TrajectoryMeasurements from DetLayer.
 *
 *  \author C. Liu, R. Bellan, N. Amapane
 *  \modified by C. Calabria to include GEMs
 *
 */

#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"

#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h" 
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Services/interface/UpdaterService.h"


typedef MuonTransientTrackingRecHit::MuonRecHitPointer MuonRecHitPointer;
typedef MuonTransientTrackingRecHit::MuonRecHitContainer MuonRecHitContainer;



MuonDetLayerMeasurements::MuonDetLayerMeasurements(edm::InputTag dtlabel, 
						   edm::InputTag csclabel, 
						   edm::InputTag rpclabel,
						   edm::InputTag gemlabel,
						   bool enableDT, bool enableCSC, bool enableRPC, bool enableGEM): 
  theDTRecHitLabel(dtlabel),
  theCSCRecHitLabel(csclabel),
  theRPCRecHitLabel(rpclabel),
  theGEMRecHitLabel(gemlabel),
  enableDTMeasurement(enableDT),
  enableCSCMeasurement(enableCSC),
  enableRPCMeasurement(enableRPC),
  enableGEMMeasurement(enableGEM),
  theDTRecHits(),
  theCSCRecHits(),
  theRPCRecHits(),
  theGEMRecHits(),
  theDTEventID(),
  theCSCEventID(),
  theRPCEventID(),
  theGEMEventID(),
  theEvent(0){
	  static int procInstance(0);
	  std::ostringstream sDT;
	  sDT<<"MuonDetLayerMeasurements::checkDTRecHits::" << procInstance;
	  theDTCheckName = sDT.str();
	  std::ostringstream sRPC;
	  sRPC<<"MuonDetLayerMeasurements::checkRPCRecHits::" << procInstance;
	  theRPCCheckName = sRPC.str();
	  std::ostringstream sCSC;
	  sCSC<<"MuonDetLayerMeasurements::checkCSCRecHits::" << procInstance;
	  theCSCCheckName = sCSC.str();
	  std::ostringstream sGEM;
	  sGEM<<"MuonDetLayerMeasurements::checkGEMRecHits::" << procInstance;
	  theGEMCheckName = sGEM.str();
	  procInstance++;
  }

MuonDetLayerMeasurements::~MuonDetLayerMeasurements(){}

MuonRecHitContainer MuonDetLayerMeasurements::recHits(const GeomDet* geomDet, 
			                              const edm::Event& iEvent)
{
  DetId geoId = geomDet->geographicalId();
  theEvent = &iEvent;
  MuonRecHitContainer result;

  if (geoId.subdetId()  == MuonSubdetId::DT) {
    if(enableDTMeasurement) 
    {
      checkDTRecHits();
    
      // Create the ChamberId
      DTChamberId chamberId(geoId.rawId());
      // LogTrace("Muon|RecoMuon|MuonDetLayerMeasurements") << "(DT): "<<chamberId<<std::endl;
    
      // Get the DT-Segment which relies on this chamber
      DTRecSegment4DCollection::range range = theDTRecHits->get(chamberId);
    
      // Create the MuonTransientTrackingRechit
      for (DTRecSegment4DCollection::const_iterator rechit = range.first; 
           rechit!=range.second;++rechit)
        result.push_back(MuonTransientTrackingRecHit::specificBuild(geomDet,&*rechit));
    }
  }
  
  else if (geoId.subdetId()  == MuonSubdetId::CSC) {
    if(enableCSCMeasurement)
    {
      checkCSCRecHits();

      // Create the chamber Id
      CSCDetId chamberId(geoId.rawId());
      //    LogTrace("Muon|RecoMuon|MuonDetLayerMeasurements") << "(CSC): "<<chamberId<<std::endl;

      // Get the CSC-Segment which relies on this chamber
      CSCSegmentCollection::range range = theCSCRecHits->get(chamberId);
    
      // Create the MuonTransientTrackingRecHit
      for (CSCSegmentCollection::const_iterator rechit = range.first; 
           rechit!=range.second; ++rechit)
        result.push_back(MuonTransientTrackingRecHit::specificBuild(geomDet,&*rechit)); 
    }
  }
  
  else if (geoId.subdetId()  == MuonSubdetId::RPC) {
    if(enableRPCMeasurement)
    {
      checkRPCRecHits(); 

      // Create the chamber Id
      RPCDetId chamberId(geoId.rawId());
      // LogTrace("Muon|RecoMuon|MuonDetLayerMeasurements") << "(RPC): "<<chamberId<<std::endl;
    
      // Get the RPC-Segment which relies on this chamber
      RPCRecHitCollection::range range = theRPCRecHits->get(chamberId);
    
      // Create the MuonTransientTrackingRecHit
      for (RPCRecHitCollection::const_iterator rechit = range.first; 
           rechit!=range.second; ++rechit)
        result.push_back(MuonTransientTrackingRecHit::specificBuild(geomDet,&*rechit));
    }
  }
  else if (geoId.subdetId()  == MuonSubdetId::GEM) {
    if(enableGEMMeasurement)
    {
      checkGEMRecHits(); 

      // Create the chamber Id
      GEMDetId chamberId(geoId.rawId());
      // LogTrace("Muon|RecoMuon|MuonDetLayerMeasurements") << "(GEM): "<<chamberId<<std::endl;
    
      // Get the GEM-Segment which relies on this chamber
      GEMRecHitCollection::range range = theGEMRecHits->get(chamberId);

      // Create the MuonTransientTrackingRecHit
      for (GEMRecHitCollection::const_iterator rechit = range.first; 
           rechit!=range.second; ++rechit)
        result.push_back(MuonTransientTrackingRecHit::specificBuild(geomDet,&*rechit));
    }
  }
  else {
    // wrong type
    throw cms::Exception("MuonDetLayerMeasurements") << "The DetLayer with det " << geoId.det() << " subdet " << geoId.subdetId() << " is not a valid Muon DetLayer. ";
  }
  return result;
}


void MuonDetLayerMeasurements::checkDTRecHits()
{
  checkEvent();
  if (!edm::Service<UpdaterService>()->checkOnce(theDTCheckName)) return;

  {
    theDTEventID = theEvent->id();
    theEvent->getByLabel(theDTRecHitLabel, theDTRecHits);
  }
  if(!theDTRecHits.isValid())
  {
    throw cms::Exception("MuonDetLayerMeasurements") << "Cannot get DT RecHits";
  }
}


void MuonDetLayerMeasurements::checkCSCRecHits()
{
  checkEvent();
  if (!edm::Service<UpdaterService>()->checkOnce(theCSCCheckName)) return;

  {
    theCSCEventID = theEvent->id();
    theEvent->getByLabel(theCSCRecHitLabel, theCSCRecHits);
  }
  if(!theCSCRecHits.isValid())
  {
    throw cms::Exception("MuonDetLayerMeasurements") << "Cannot get CSC RecHits";
  }
}


void MuonDetLayerMeasurements::checkRPCRecHits()
{
  checkEvent();
  if (!edm::Service<UpdaterService>()->checkOnce(theRPCCheckName)) return;

  {
    theRPCEventID = theEvent->id();
    theEvent->getByLabel(theRPCRecHitLabel, theRPCRecHits);
  }
  if(!theRPCRecHits.isValid())
  {
    throw cms::Exception("MuonDetLayerMeasurements") << "Cannot get RPC RecHits";
  }
}


void MuonDetLayerMeasurements::checkGEMRecHits()
{
  checkEvent();
  if (!edm::Service<UpdaterService>()->checkOnce(theGEMCheckName)) return;

  {
    theGEMEventID = theEvent->id();
    theEvent->getByLabel(theGEMRecHitLabel, theGEMRecHits);
  }
  if(!theGEMRecHits.isValid())
  {
    throw cms::Exception("MuonDetLayerMeasurements") << "Cannot get GEM RecHits";
  }
}


///measurements method if already got the Event 
MeasurementContainer
MuonDetLayerMeasurements::measurements( const DetLayer* layer,
					const TrajectoryStateOnSurface& startingState,
					const Propagator& prop,
					const MeasurementEstimator& est) {
  checkEvent();
  return measurements(layer, startingState, prop, est, *theEvent);
}


MeasurementContainer
MuonDetLayerMeasurements::measurements(const DetLayer* layer,
				       const TrajectoryStateOnSurface& startingState,
				       const Propagator& prop,
				       const MeasurementEstimator& est,
				       const edm::Event& iEvent) {
  
  MeasurementContainer result;
  
  std::vector<DetWithState> dss = layer->compatibleDets(startingState, prop, est);
  LogTrace("RecoMuon")<<"compatibleDets: "<<dss.size()<<std::endl;
  
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
MuonDetLayerMeasurements::measurements( const DetLayer* layer,
					const GeomDet* det,
					const TrajectoryStateOnSurface& stateOnDet,
					const MeasurementEstimator& est,
					const edm::Event& iEvent) {
  MeasurementContainer result;
  
  // Get the Segments which relies on the GeomDet given by compatibleDets
  MuonRecHitContainer muonRecHits = recHits(det, iEvent);
  
  // Create the Trajectory Measurement
  for(MuonRecHitContainer::const_iterator rechit = muonRecHits.begin();
      rechit != muonRecHits.end(); ++rechit) {

    MeasurementEstimator::HitReturnType estimate = est.estimate(stateOnDet,**rechit);
    LogTrace("RecoMuon")<<"Dimension: "<<(*rechit)->dimension()
			<<" Chi2: "<<estimate.second<<std::endl;
    if (estimate.first) {
      result.push_back(TrajectoryMeasurement(stateOnDet, rechit->get(),
					     estimate.second,layer));
    }
  }

  if (!result.empty()) sort( result.begin(), result.end(), TrajMeasLessEstim());
   
  return result;
}



MeasurementContainer
MuonDetLayerMeasurements::fastMeasurements( const DetLayer* layer,
					    const TrajectoryStateOnSurface& theStateOnDet,
					    const TrajectoryStateOnSurface& startingState,
					    const Propagator& prop,
					    const MeasurementEstimator& est,
					    const edm::Event& iEvent) {
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

///fastMeasurements method if already got the Event
MeasurementContainer
MuonDetLayerMeasurements::fastMeasurements(const DetLayer* layer,
					   const TrajectoryStateOnSurface& theStateOnDet,
					   const TrajectoryStateOnSurface& startingState,
					   const Propagator& prop,
					   const MeasurementEstimator& est) {
  checkEvent();
  return fastMeasurements(layer, theStateOnDet, startingState, prop, est, *theEvent); 
}


std::vector<TrajectoryMeasurementGroup>
MuonDetLayerMeasurements::groupedMeasurements(const DetLayer* layer,
					      const TrajectoryStateOnSurface& startingState,
					      const Propagator& prop,
					      const MeasurementEstimator& est) {
  checkEvent();
  return groupedMeasurements(layer, startingState, prop,  est, *theEvent);
}


std::vector<TrajectoryMeasurementGroup>
MuonDetLayerMeasurements::groupedMeasurements(const DetLayer* layer,
					      const TrajectoryStateOnSurface& startingState,
					      const Propagator& prop,
					      const MeasurementEstimator& est,
					      const edm::Event& iEvent) {
  
  std::vector<TrajectoryMeasurementGroup> result;
  // if we want to use the concept of InvalidRecHits,
  // we can reuse LayerMeasurements from TrackingTools/MeasurementDet
  std::vector<DetGroup> groups(layer->groupedCompatibleDets(startingState, prop, est));

  // this should be fixed either in RecoMuon/MeasurementDet/MuonDetLayerMeasurements or
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
void MuonDetLayerMeasurements::setEvent(const edm::Event& event) {
  theEvent = &event;
}


void MuonDetLayerMeasurements::checkEvent() const {
  if(!theEvent)
    throw cms::Exception("MuonDetLayerMeasurements") << "The event has not been set";
}

MuonRecHitContainer MuonDetLayerMeasurements::recHits(const DetLayer* layer, 
						      const edm::Event& iEvent) {
  MuonRecHitContainer rhs;
  
  std::vector <const GeomDet*> gds = layer->basicComponents();

  for (std::vector<const GeomDet*>::const_iterator igd = gds.begin(); 
       igd != gds.end(); igd++) {
    MuonRecHitContainer detHits = recHits(*igd, iEvent);
    rhs.insert(rhs.end(), detHits.begin(), detHits.end());
  }
  return rhs;
}

MuonRecHitContainer MuonDetLayerMeasurements::recHits(const DetLayer* layer) 
{
  checkEvent();
  return recHits(layer, *theEvent);
}

