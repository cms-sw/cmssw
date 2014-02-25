/**
 *  Class: DynamicTruncation
 *
 *  Description:
 *  class for the dynamical stop of the KF according to the
 *  compatibility degree between the extrapolated track
 *  state and the reconstructed segment in the muon chambers
 *
 *  $Date: 2014/02/18 08:50:12 $
 *  $Revision: 1.11 $
 *
 *  Authors :
 *  D. Pagano & G. Bruno - UCL Louvain
 *
 **/

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoMuon/GlobalTrackingTools/interface/DynamicTruncation.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "TrackingTools/TrackFitters/interface/RecHitLessByDet.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"
#include "RecoMuon/Navigation/interface/MuonNavigationSchool.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "RecoMuon/Navigation/interface/MuonNavigationPrinter.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"

#define MAX_THR 1e7

using namespace edm;
using namespace std;
using namespace reco;



DynamicTruncation::DynamicTruncation(const edm::Event& event, const MuonServiceProxy& theService) {
  propagator = theService.propagator("SmartPropagator");
  propagatorPF = theService.propagator("SmartPropagatorOpposite");
  propagatorCompatibleDet = theService.propagator("SteppingHelixPropagatorAny");
  theG = theService.trackingGeometry();
  theService.eventSetup().get<TransientRecHitRecord>().get("MuonRecHitBuilder",theMuonRecHitBuilder);
  theService.eventSetup().get<TrackingComponentsRecord>().get("KFUpdator",updatorHandle);
  theService.eventSetup().get<MuonGeometryRecord>().get(cscGeom);
  theService.eventSetup().get<MuonRecoGeometryRecord>().get(navMuon);
  theService.eventSetup().get<IdealMagneticFieldRecord>().get(magfield);
  navigation = new DirectMuonNavigation(theService.detLayerGeometry());
  doUpdateOfKFStates = true;
  
  // Initialize utilities for segments
  getSegs = new ChamberSegmentUtility(event, theService.eventSetup());

  // Initialize utilities for thresholds
  thrManager = new ThrParameters(&theService.eventSetup());
  useDBforThr = thrManager->isValidThdDB();
  if (useDBforThr) dytThresholds = thrManager->getInitialThresholds();

  if (useDBforThr) cout << "Thresholds will be taken from DB.\n";
  else cout << "Thresholds will be taken from cfg file.\n";
}

DynamicTruncation::~DynamicTruncation() {
  delete navigation;
  delete thrManager;
  delete getSegs;
}

void DynamicTruncation::update(TrajectoryStateOnSurface& tsos, ConstRecHitPointer rechit) {
  TrajectoryStateOnSurface temp = updatorHandle->update(tsos, *rechit);
  if (temp.isValid()) tsos = updatorHandle->update(tsos, *rechit);
}



/////////////////////////////////
///// Configuration methods /////
/////////////////////////////////
void DynamicTruncation::setSelector(int selector) {
  if (selector < 0 || selector > 2) throw cms::Exception("NotAvailable") << "DYT selector: wrong option!" << endl;
  cout << "DYT selector set to " << selector << ".\n";
  if (selector == 0) cout << "[DYT disabled]\n";
  if (selector == 1) cout << "[use all compatible stations]\n";
  if (selector == 2) cout << "[stop at second consecutive incompatible station]\n";
  DYTselector = selector;

}

void DynamicTruncation::setUseAPE(bool useAPE_) {
  if (!useAPE_) cout << "The use of APE is disabled." << endl;
  useAPE = useAPE_;
}

void DynamicTruncation::setUpdateState(bool upState) {
  if (!upState) cout << "KF state updator disabled." << endl;
  doUpdateOfKFStates = upState;
}

void DynamicTruncation::setThr(const std::vector<int>& thr) {
  if (thr.size() == 2) {
    for (unsigned int i = 0; i < thr.size(); i++)
      if (thr[i] >= 0) Thrs.push_back(thr[i]);
      else Thrs.push_back(MAX_THR);
    return;
  }
  throw cms::Exception("NotAvailable") << "WARNING: wrong size for the threshold vector!\nExpected size: 2\n   Found size: " << thr.size();
}
/////////////////////////////////
/////////////////////////////////
/////////////////////////////////



//===> filter
TransientTrackingRecHit::ConstRecHitContainer DynamicTruncation::filter(const Trajectory& traj) {
  result.clear();
  prelFitMeas.clear();
  
  // Get APE maps 
  dtApeMap = thrManager->GetDTApeMap();
  cscApeMap = thrManager->GetCSCApeMap();

  // Get Last tracker TSOS (updated)
  std::vector<TrajectoryMeasurement> muonMeasurements = traj.measurements();
  TrajectoryMeasurement lastTKm = muonMeasurements.front();
  for (std::vector<TrajectoryMeasurement>::const_iterator imT = muonMeasurements.begin(); imT != muonMeasurements.end(); imT++ ) {
    if ( !(*imT).recHit()->isValid() ) continue;
    const TransientTrackingRecHit* hit = &(*(*imT).recHit());
    if (hit->geographicalId().det() == DetId::Tracker) {
      result.push_back((*imT).recHit());
      if (!(*imT).forwardPredictedState().isValid()) continue;
      if ((*imT).forwardPredictedState().globalPosition().mag() >
	  lastTKm.forwardPredictedState().globalPosition().mag()) lastTKm = *imT;
    }
  }
  currentState = lastTKm.forwardPredictedState();
  update(currentState, lastTKm.recHit());

  prelFitState = lastTKm.forwardPredictedState();
  update(prelFitState, lastTKm.recHit());
  prelFitMeas = result;

  // Run the DYT
  filteringAlgo();

  return result;
}


//===> filteringAlgo
void DynamicTruncation::filteringAlgo() {
  map<int, vector<DetId> > compatibleIds;
  map<int, vector<DTRecSegment4D> > dtSegMap;
  map<int, vector<CSCSegment> > cscSegMap;
  int incompConLay = 0;
  nStationsUsed = 0;

  // Get list of compatible layers
  compatibleDets(currentState, compatibleIds);

  // Fill segment maps
  fillSegmentMaps(compatibleIds, dtSegMap, cscSegMap); 

  // Do a preliminary fit
  if (useDBforThr) preliminaryFit(compatibleIds, dtSegMap, cscSegMap);

  // Loop on compatible layers
  for (map<int, vector<DetId> >::iterator it=compatibleIds.begin(); it!=compatibleIds.end(); ++it) {
    int stLayer = stationfromDet(it->second.front());
    DTRecSegment4D bestDTSeg;
    CSCSegment bestCSCSeg;
    double bestDTEstimator  = MAX_THR;
    double bestCSCEstimator = MAX_THR;
    vector<DTRecSegment4D> dtSegs   = dtSegMap[it->first];
    vector<CSCSegment> cscSegs      = cscSegMap[it->first];
    

    /////////////////
    // Debug stuff //
    /////////////////
    if (it->second.front().subdetId() == MuonSubdetId::DT) {
      DTChamberId ch(it->second.front());
      cout << ch << " - DT segs: " << dtSegs.size() << " - CSC segs: " << cscSegs.size() << endl;
    } else {
      CSCDetId ch(it->second.front());
      cout << ch << " - DT segs: " << dtSegs.size() << " - CSC segs: " << cscSegs.size() << endl;
    }
    /////////////////
    /////////////////
    /////////////////


    // DT case: find the most compatible segment 
    TrajectoryStateOnSurface tsosDTlayer;
    testDTstation(currentState, dtSegs, bestDTEstimator, bestDTSeg, tsosDTlayer);

    // CSC case: find the most compatible segment
    TrajectoryStateOnSurface tsosCSClayer;
    testCSCstation(currentState, cscSegs, bestCSCEstimator, bestCSCSeg, tsosCSClayer);

    // Decide whether to keep the layer or not
    bool chosenLayer = chooseLayers(incompConLay, bestDTEstimator, bestDTSeg, tsosDTlayer, bestCSCEstimator, bestCSCSeg, tsosCSClayer);
    fillDYTInfos(stLayer, chosenLayer, incompConLay, bestDTEstimator, bestCSCEstimator, bestDTSeg, bestCSCSeg); 
  }
  cout << "Number of used stations = " << nStationsUsed << endl;
}


//===> stationfromDet
int DynamicTruncation::stationfromDet(DetId const& det) {
  if (det.subdetId() == MuonSubdetId::CSC) {
    CSCDetId ch(det);
    return ch.station();
  }
  if (det.subdetId() == MuonSubdetId::DT) {
    DTChamberId ch(det);
    return ch.station();
  }
  return 0;
}


//===> fillDYTInfos
void DynamicTruncation::fillDYTInfos(int const &st, bool const &chosenLayer, int &incompConLay, 
				     double const &bestDTEstimator, double const &bestCSCEstimator,
				     DTRecSegment4D const &bestDTSeg, CSCSegment const &bestCSCSeg) {
  if (chosenLayer) {
    nStationsUsed++;
    incompConLay = 0;
    if (bestDTEstimator <= bestCSCEstimator) {
      estimatorMap[st] = bestDTEstimator;
      DetId id(bestDTSeg.chamberId()); 
      idChamberMap[st] = id;
    } else {
      DetId id(bestCSCSeg.cscDetId());
      idChamberMap[st] = id;
      estimatorMap[st] = bestCSCEstimator;
    }
    usedStationMap[st] = true;
  } else {
    incompConLay++;
    estimatorMap[st] = -1;
    usedStationMap[st] = false;
  }
}


//===> compatibleDets
void DynamicTruncation::compatibleDets(TrajectoryStateOnSurface &tsos, map<int, std::vector<DetId> > &detMap) {
  MuonPatternRecoDumper dumper;
  MeasurementEstimator *theEstimator = new Chi2MeasurementEstimator(1000, 1000);
  std::vector<const DetLayer *> navLayers;
  navLayers = navigation->compatibleLayers(*(currentState.freeState()), alongMomentum);
  unsigned int ilayerCorrected = 0;
  for ( unsigned int ilayer=0; ilayer<navLayers.size(); ilayer++ ) {
    // Skip RPC layers
    if (navLayers[ilayer]->subDetector() != GeomDetEnumerators::DT &&
	navLayers[ilayer]->subDetector() != GeomDetEnumerators::CSC) continue;
    ilayerCorrected++;
    std::vector<DetLayer::DetWithState> comps = navLayers[ilayer]->compatibleDets(currentState, *propagatorCompatibleDet, *theEstimator);
    cout << comps.size() << " compatible Dets with " << navLayers[ilayer]->subDetector() << " Layer " << ilayer << " " 
	 << dumper.dumpLayer(navLayers[ilayer]);
    if (comps.size() > 0) {
      for ( unsigned int icomp=0; icomp<comps.size(); icomp++ ) {
		DetId id(comps[icomp].first->geographicalId().rawId());
		detMap[ilayerCorrected].push_back(id);
      }
    }
  }
  if (theEstimator) delete theEstimator;
}


//===> fillSegmentMaps
void DynamicTruncation::fillSegmentMaps( map<int, vector<DetId> > &compatibleIds,
                                         map<int, vector<DTRecSegment4D> > &dtSegMap,
                                         map<int, vector<CSCSegment> > &cscSegMap) {
  for (map<int, vector<DetId> >::iterator it=compatibleIds.begin(); it!=compatibleIds.end(); ++it) {
    std::vector<DetId> ids = compatibleIds[it->first];
    for (unsigned j = 0; j < ids.size(); j++) {
      if (ids[j].subdetId() == MuonSubdetId::CSC) {
        CSCDetId ch(ids[j]);
        vector<CSCSegment> tmp = getSegs->getCSCSegmentsInChamber(ch);
        for (unsigned int k = 0; k < tmp.size(); k++) cscSegMap[it->first].push_back(tmp[k]);
      }
      if (ids[j].subdetId() == MuonSubdetId::DT) {
        DTChamberId ch(ids[j]);
        vector<DTRecSegment4D> tmp = getSegs->getDTSegmentsInChamber(ch);
        for (unsigned int k = 0; k < tmp.size(); k++) dtSegMap[it->first].push_back(tmp[k]);
      }
    }
  }
}


//===> testDTstation
void DynamicTruncation::testDTstation(TrajectoryStateOnSurface &startingState, vector<DTRecSegment4D> const &segments, 
				      double &bestEstimator, DTRecSegment4D &bestSeg, TrajectoryStateOnSurface &tsosdt) {
  if (segments.size() == 0) return;
  for (unsigned int iSeg = 0; iSeg < segments.size(); iSeg++) {
    DTChamberId chamber(segments[iSeg].chamberId());
    tsosdt = propagator->propagate(startingState, theG->idToDet(chamber)->surface());
    if (!tsosdt.isValid()) continue;
    LocalError apeLoc;
    if (useAPE) apeLoc = ErrorFrameTransformer().transform(dtApeMap.find(chamber)->second, theG->idToDet(chamber)->surface());
    StateSegmentMatcher estim(tsosdt, segments[iSeg], apeLoc);
    double estimator = estim.value();      
    cout << "estimator DT = " << estimator << endl;
    if (estimator >= bestEstimator) continue; 
    bestEstimator = estimator;
    bestSeg = segments[iSeg];
  }
}


//===> testCSCstation
void DynamicTruncation::testCSCstation(TrajectoryStateOnSurface &startingState, vector<CSCSegment> const &segments, 
				       double &bestEstimator, CSCSegment &bestSeg, TrajectoryStateOnSurface &tsoscsc) {
  if (segments.size() == 0) return;
  for (unsigned int iSeg = 0; iSeg < segments.size(); iSeg++) {
    CSCDetId chamber(segments[iSeg].cscDetId());
    tsoscsc = propagator->propagate(startingState, theG->idToDet(chamber)->surface());
    if (!tsoscsc.isValid()) continue;
    LocalError apeLoc;
    if (useAPE) apeLoc = ErrorFrameTransformer().transform(cscApeMap.find(chamber)->second, theG->idToDet(chamber)->surface());
    StateSegmentMatcher estim(tsoscsc, segments[iSeg], apeLoc);
    double estimator = estim.value();
    cout << "estimator CSC = " << estimator << endl;
    if (estimator >= bestEstimator) continue;
    bestEstimator = estimator;
    bestSeg = segments[iSeg]; 
  }
}


//===> useSegment
void DynamicTruncation::useSegment(DTRecSegment4D const &bestDTSeg, TrajectoryStateOnSurface const &tsosDT) {
  result.push_back(theMuonRecHitBuilder->build(&bestDTSeg));
  if (doUpdateOfKFStates) currentState = updatorHandle->update(tsosDT, *theMuonRecHitBuilder->build(&bestDTSeg));
  else currentState = tsosDT;
}


//===> useSegment 
void DynamicTruncation::useSegment(CSCSegment const &bestCSCSeg, TrajectoryStateOnSurface const &tsosCSC) {
  result.push_back(theMuonRecHitBuilder->build(&bestCSCSeg));
  if (doUpdateOfKFStates) currentState = updatorHandle->update(tsosCSC, *theMuonRecHitBuilder->build(&bestCSCSeg));
  else currentState = tsosCSC;
}


//===> preliminaryFit
void DynamicTruncation::preliminaryFit(map<int, vector<DetId> > compatibleIds, map<int, vector<DTRecSegment4D> > dtSegMap,
				       map<int, vector<CSCSegment> > cscSegMap) {
  for (map<int, vector<DetId> >::iterator it=compatibleIds.begin(); it!=compatibleIds.end(); ++it) {
    DTRecSegment4D bestDTSeg;
    CSCSegment bestCSCSeg;
    double bestDTEstimator  = MAX_THR;
    double bestCSCEstimator = MAX_THR;
    double initThr = MAX_THR;
    vector<DTRecSegment4D> dtSegs = dtSegMap[it->first];
    vector<CSCSegment> cscSegs    = cscSegMap[it->first];

    // DT case: find the most compatible segment
    TrajectoryStateOnSurface tsosDTlayer;
    testDTstation(prelFitState, dtSegs, bestDTEstimator, bestDTSeg, tsosDTlayer);

    // CSC case: find the most compatible segment
    TrajectoryStateOnSurface tsosCSClayer;
    testCSCstation(prelFitState, cscSegs, bestCSCEstimator, bestCSCSeg, tsosCSClayer);

    // Decide whether to keep the layer or not
    if (bestDTEstimator == MAX_THR && bestCSCEstimator == MAX_THR) continue;
    if (bestDTEstimator <= bestCSCEstimator) {
      getThresholdFromCFG(initThr, DetId(bestDTSeg.chamberId()));
      if (bestDTEstimator >= initThr) continue;
      prelFitMeas.push_back(theMuonRecHitBuilder->build(&bestDTSeg));
      prelFitState = updatorHandle->update(tsosDTlayer, *theMuonRecHitBuilder->build(&bestDTSeg));
    } else {
      getThresholdFromCFG(initThr, DetId(bestCSCSeg.cscDetId()));
      if (bestCSCEstimator >= initThr) continue;
      prelFitMeas.push_back(theMuonRecHitBuilder->build(&bestCSCSeg));
      prelFitState = updatorHandle->update(tsosCSClayer, *theMuonRecHitBuilder->build(&bestCSCSeg));
    }
  }
  prelFitMeas.pop_back();
  for (ConstRecHitContainer::const_iterator imrh = prelFitMeas.end(); imrh != prelFitMeas.begin(); imrh-- ) {
    DetId id = (*imrh)->geographicalId(); 
    TrajectoryStateOnSurface tmp = propagatorPF->propagate(prelFitState, theG->idToDet(id)->surface());
    if (tmp.isValid()) prelFitState = tmp; 
  }
  muonPTest  = prelFitState.globalMomentum().perp();
  muonETAest = prelFitState.globalMomentum().eta();
}


//===> chooseLayers
bool DynamicTruncation::chooseLayers(int &incompLayers, double const &bestDTEstimator, DTRecSegment4D const &bestDTSeg, TrajectoryStateOnSurface const &tsosDT, 
				     double const &bestCSCEstimator, CSCSegment const &bestCSCSeg, TrajectoryStateOnSurface const &tsosCSC) {
  double initThr = MAX_THR;
  if (bestDTEstimator == MAX_THR && bestCSCEstimator == MAX_THR) return false;

  if (bestDTEstimator <= bestCSCEstimator) {
    // Get threshold for the chamber
    if (useDBforThr) getThresholdFromDB(initThr, DetId(bestDTSeg.chamberId()));
    else getThresholdFromCFG(initThr, DetId(bestDTSeg.chamberId())); 
    cout << "Threshold: " << initThr << endl;
    if (DYTselector == 0) {useSegment(bestDTSeg, tsosDT); return true;}
    if (DYTselector == 1 && bestDTEstimator < initThr) {useSegment(bestDTSeg, tsosDT); return true;}
    if (DYTselector == 2 && incompLayers < 2 && bestDTEstimator < initThr) {useSegment(bestDTSeg, tsosDT); return true;}
  } else {
    // Get threshold for the chamber
    if (useDBforThr) getThresholdFromDB(initThr, DetId(bestCSCSeg.cscDetId()));
    else getThresholdFromCFG(initThr, DetId(bestCSCSeg.cscDetId()));
    cout << "Threshold: " << initThr << endl;
    if (DYTselector == 0) {useSegment(bestCSCSeg, tsosCSC); return true;}
    if (DYTselector == 1 && bestCSCEstimator < initThr) {useSegment(bestCSCSeg, tsosCSC); return true;}
    if (DYTselector == 2 && incompLayers < 2 && bestCSCEstimator < initThr) {useSegment(bestCSCSeg, tsosCSC); return true;}
  }
  return false;
}


//===> getThresholdFromDB
void DynamicTruncation::getThresholdFromDB(double& thr, DetId const& id) {
  vector<DYTThrObject::dytThrStruct> thrvector = dytThresholds->thrsVec;
  for (vector<DYTThrObject::dytThrStruct>::const_iterator it = thrvector.begin(); it != thrvector.end(); it++) {
    DYTThrObject::dytThrStruct obj = (*it);
    if (obj.id == id) {
      thr = obj.thr;
      break;
    }
  }
  cout << "thr before pt = " << thr << endl;
  correctThrByPtAndEta(thr);
  cout << "thr after pt = " << thr << endl;
}


//===> correctThrByPtAndEta
void DynamicTruncation::correctThrByPtAndEta(double& thr) {
  //muonPTest 
  if (fabs(muonETAest) <= 0.8) thr *= 0.5;
  if (fabs(muonETAest) > 0.8 && fabs(muonETAest) <= 1.2) thr *= 0.5;
  if (fabs(muonETAest) > 1.2 && fabs(muonETAest) <= 1.6) thr *= 0.5;
  if (fabs(muonETAest) > 1.6 && fabs(muonETAest) <= 2.0) thr *= 0.5;
  if (fabs(muonETAest) > 2.0) thr *= 0.5;
}


//===> getThresholdFromCFG
void DynamicTruncation::getThresholdFromCFG(double& thr, DetId const& id) {
  if (id.subdetId() == MuonSubdetId::DT) {
    thr = Thrs[0];
  }
  if (id.subdetId() == MuonSubdetId::CSC) {
    thr = Thrs[1];
  }
}
