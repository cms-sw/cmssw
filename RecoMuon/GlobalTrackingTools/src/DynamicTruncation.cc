/**
 *  Class: DynamicTruncation
 *
 *  Description:
 *  class for the dynamical stop of the KF according to the
 *  compatibility degree between the extrapolated track
 *  state and the reconstructed segment in the muon chambers
 *
 *  $Date: 2010/06/14 11:02:21 $
 *  $Revision: 1.1 $
 *
 *  Authors :
 *  D. Pagano & G. Bruno - UCL Louvain
 *
 **/

#include "RecoMuon/GlobalTrackingTools/interface/DynamicTruncation.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "TrackingTools/TrackFitters/interface/RecHitLessByDet.h"
#include "RecoMuon/GlobalTrackingTools/interface/ChamberSegmentUtility.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"
#include "RecoMuon/Navigation/interface/MuonNavigationSchool.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "RecoMuon/Navigation/interface/MuonNavigationPrinter.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"
#include "RecoMuon/GlobalTrackingTools/interface/MuonTkNavigationSchool.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"


#define MAX_THR 1e7


DynamicTruncation::DynamicTruncation(const edm::Event& event, const MuonServiceProxy& theService) {
  theEvent = &event;
  theSetup = &theService.eventSetup();
  propagator = theService.propagator("SmartPropagator");
  //propagator = theService.propagator("SteppingHelixPropagatorAny");
  theG = theService.trackingGeometry();
  theService.eventSetup().get<TransientRecHitRecord>().get("MuonRecHitBuilder",theMuonRecHitBuilder);
  theService.eventSetup().get<TrackingComponentsRecord>().get("KFUpdator",updatorHandle);
  theService.eventSetup().get<MuonGeometryRecord>().get(cscGeom);
  theService.eventSetup().get<MuonRecoGeometryRecord>().get(navMuon);
  theService.eventSetup().get<IdealMagneticFieldRecord>().get(magfield);
  navigation = new DirectMuonNavigation(theService.detLayerGeometry());
}



DynamicTruncation::~DynamicTruncation() {
}



TransientTrackingRecHit::ConstRecHitContainer DynamicTruncation::filter(const Trajectory& traj) {

  // Put the tracker hits in the final vector and get the last tracker valid measure
  vector<TrajectoryMeasurement> muonMeasurements = traj.measurements();
  TrajectoryMeasurement lastTKm = muonMeasurements.front();
  for (std::vector<TrajectoryMeasurement>::const_iterator imT = muonMeasurements.begin(); imT != muonMeasurements.end(); imT++ ) {
    if ( !(*imT).recHit()->isValid() ) continue;
    const TransientTrackingRecHit* hit = &(*(*imT).recHit());
    if (hit->geographicalId().det() == DetId::Tracker) {
      result.push_back((*imT).recHit());
      if (!(*imT).forwardPredictedState().isValid()) continue;
      if ((*imT).forwardPredictedState().globalPosition().mag() > lastTKm.forwardPredictedState().globalPosition().mag()) lastTKm = *imT;
    }
  }

  // get the last (forward) predicted state for the tracker
  currentState = lastTKm.forwardPredictedState();
  
  // update the state with the last tracker measure
  update(currentState, lastTKm.recHit());

  // use the navigation to get the list of compatible dets
  map<int, vector<DetId> > detMap;
  compatibleDets(currentState, detMap);
  
  // selects the muon hits for the final refit
  filteringAlgo(detMap);

  return result;
}


 
void DynamicTruncation::setThr(int bThr, int eThr) {
  if (bThr <= MAX_THR && bThr >= 0) DTThr  = bThr; // DT thr 
  else DTThr = MAX_THR;
  if (eThr <= MAX_THR && eThr >= 0) CSCThr = eThr; // CSC thr
  else CSCThr = MAX_THR;
}



double DynamicTruncation::getBest(vector<CSCSegment>& segs, TrajectoryStateOnSurface& tsos, CSCSegment& bestCSCSeg) {
  unsigned int i;
  double val = MAX_THR;
  vector<CSCSegment>::size_type sz = segs.size();
  for (i=0; i<sz; i++) {
    StateSegmentMatcher estim(&tsos, &segs[i]);
    double tmp = estim.value();
    if (tmp < val) {
      bestCSCSeg = segs[i];
      val = tmp;
    }
  }
  return val;
}



double DynamicTruncation::getBest(vector<DTRecSegment4D>& segs, TrajectoryStateOnSurface& tsos, DTRecSegment4D& bestDTSeg) {
  unsigned int i;
  double val = MAX_THR;
  vector<DTRecSegment4D>::size_type sz = segs.size();
  for (i=0; i<sz; i++) {
    StateSegmentMatcher estim(&tsos, &segs[i]); 
    double tmp = estim.value();                                                                                                                                              
    if (tmp < val) {
      bestDTSeg = segs[i];
      val = tmp;
    }
  }
  return val;
}



void DynamicTruncation::compatibleDets(TrajectoryStateOnSurface& tsos, map<int, vector<DetId> >& detMap) {
  SteppingHelixPropagator prop(magfield.product(), anyDirection);
  MuonPatternRecoDumper dumper;
  MeasurementEstimator *theEstimator = new Chi2MeasurementEstimator(100, 3);
  vector<const DetLayer *> navLayers;
  navLayers = navigation->compatibleLayers(*(currentState.freeState()), alongMomentum);
  unsigned int nlayer = 0;
  for ( unsigned int ilayer=0; ilayer<navLayers.size(); ilayer++ ) {
    // Skip RPC layers
    if (navLayers[ilayer]->subDetector() == GeomDetEnumerators::RPCEndcap || navLayers[ilayer]->subDetector() == GeomDetEnumerators::RPCBarrel) continue;
    vector<DetLayer::DetWithState> comps = navLayers[ilayer]->compatibleDets(currentState, prop, *theEstimator);
    //    cout << comps.size() << " compatible Dets with " << navLayers[ilayer]->subDetector() << " Layer " << ilayer << " "
    //    	 << dumper.dumpLayer(navLayers[ilayer]) << " " << endl;
    if (comps.size() > 0) {
      DetId id(comps.front().first->geographicalId().rawId());
      detMap[nlayer].push_back(id);
    }
  }
}



void DynamicTruncation::filteringAlgo(map<int, vector<DetId> >& detMap) {
  ChamberSegmentUtility getSegs(*theEvent, *theSetup);
  for (unsigned int i = 0; i < detMap.size(); ++i) {
    double bestLayerValue = MAX_THR;
    ConstRecHitContainer layerRH;
    vector<DetId> chamber = detMap[i];
    for (unsigned int j = 0; j < chamber.size(); j++) {
      DetId id = chamber[j];
      
      if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::DT) {
        DTChamberId DTid(id);
	
	vector<DTRecSegment4D> allDTsegs;
	try {allDTsegs = getSegs.getDTlist().find(DTid.station())->second;} catch (...) {;} 
	
	vector<DTRecSegment4D>::size_type sz = allDTsegs.size();
	for (i=0; i<sz; i++) {
	  
	  // Propagate the state to the current chamber
	  TrajectoryStateOnSurface tsosdt = propagator->propagate(currentState, theG->idToDet(allDTsegs[i].chamberId())->surface());
	  if (!tsosdt.isValid()) continue;

	  vector<DTRecSegment4D> DTsegs;
	  DTsegs.push_back(allDTsegs[i]); 
	  
	  DTRecSegment4D bestDTSeg;
	  double bestChamberValue = getBest(DTsegs, tsosdt, bestDTSeg);
	  if (bestChamberValue < bestLayerValue) bestLayerValue = bestChamberValue;
	  
	  // Check if the best estimator value is below the THR and then get the RH componing the segment
	  if (bestChamberValue >= DTThr || bestChamberValue > bestLayerValue) continue; 
	  layerRH.clear();
	  vector<DTRecHit1D> DTrh = getSegs.getDTRHmap(bestDTSeg);
	  for (vector<DTRecHit1D>::iterator it = DTrh.begin(); it != DTrh.end(); it++) {
	    layerRH.push_back(theMuonRecHitBuilder->build(&*it));
	  }
	}
      }
      if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::CSC) {
        CSCDetId CSCid(id);
	
	vector<CSCSegment> allCSCsegs;
	try {allCSCsegs = getSegs.getCSClist().find(CSCid.station())->second;} catch (...) {;} 
	
	vector<CSCSegment>::size_type sz = allCSCsegs.size();
	for (i=0; i<sz; i++) {
	  
	  // Propagate the state to the current chamber
	  TrajectoryStateOnSurface tsoscsc = propagator->propagate(currentState, theG->idToDet(allCSCsegs[i].cscDetId())->surface());
	  if (!tsoscsc.isValid()) continue;
	  
	  vector<CSCSegment> CSCsegs;
	  CSCsegs.push_back(allCSCsegs[i]);
	  
	  CSCSegment bestCSCSeg;
	  double bestChamberValue = getBest(CSCsegs, tsoscsc, bestCSCSeg);
	  if (bestChamberValue < bestLayerValue) bestLayerValue = bestChamberValue;
	  
	  // Check if the best estimator value is below the THR and then get the RH componing the segment
	  if (bestChamberValue >= CSCThr || bestChamberValue > bestLayerValue) continue;
	  layerRH.clear();
	  
	  vector<CSCRecHit2D> CSCrh = getSegs.getCSCRHmap(bestCSCSeg);
	  for (vector<CSCRecHit2D>::iterator it = CSCrh.begin(); it != CSCrh.end(); ++it) {
	    layerRH.push_back(theMuonRecHitBuilder->build(&*it));
	  }
	}
      }
    }
    
    if (layerRH.size() > 0) {                                                                                                                                          
      for (ConstRecHitContainer::iterator it = layerRH.begin(); it != layerRH.end(); ++it) {                                                                         
	result.push_back((*it));                                                                                                                                         
      }                                                                                                                                                                
      
      // sort the vector                                                                                                                                             
      layerRH = sort(layerRH);                                                                                                                                         
      
      // update the currentState using all rh                                                                                                             
      DetId id = layerRH.front()->geographicalId();                                                                                                      
      if (id.subdetId() == MuonSubdetId::DT) updateWithDThits(layerRH);                                                                                 
      else updateWithCSChits(layerRH);                                                                                                                 
    }
    layerRH.clear();                                                                                                                                                    
  }
}  



void DynamicTruncation::update(TrajectoryStateOnSurface& tsos, ConstRecHitPointer rechit) {
  TrajectoryStateOnSurface temp = updatorHandle->update(tsos, *rechit);
  if (temp.isValid()) currentState = updatorHandle->update(tsos, *rechit);
}



void DynamicTruncation::updateWithDThits(ConstRecHitContainer& recHits) {
  for (ConstRecHitContainer::const_iterator it = recHits.begin(); it != recHits.end(); ++it) {
    DTLayerId layid((*it)->det()->geographicalId());
    TrajectoryStateOnSurface temp = propagator->propagate(currentState, theG->idToDet(layid)->surface());
    if (temp.isValid()) currentState = updatorHandle->update(temp, **it);
  }
}



void DynamicTruncation::updateWithCSChits(ConstRecHitContainer& recHits) {
  for (ConstRecHitContainer::const_iterator it = recHits.begin(); it != recHits.end(); ++it) {
    const CSCLayer* cscChamber = cscGeom->layer((*it)->det()->geographicalId());
    CSCDetId layid = cscChamber->geographicalId();
    TrajectoryStateOnSurface temp = propagator->propagate(currentState, theG->idToDet(layid)->surface());
    if (temp.isValid()) currentState = updatorHandle->update(temp, **it);
  }
}



TransientTrackingRecHit::ConstRecHitContainer DynamicTruncation::sort(ConstRecHitContainer& recHits) {
  unsigned int i,j;
  ConstRecHitContainer::size_type n = recHits.size();
  for(i=1; i<n; ++i)
    for(j=n-1; j>=i; --j)
      {
	if(recHits[j-1]->globalPosition().mag() > recHits[j]->globalPosition().mag())
	  {
	    swap (recHits[j-1],recHits[j]);
	  }
      }
  return recHits;
}

