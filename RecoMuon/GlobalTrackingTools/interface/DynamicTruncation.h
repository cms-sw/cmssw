#ifndef RecoMuon_GlobalTrackingTools_DynamicTruncation_h
#define RecoMuon_GlobalTrackingTools_DynamicTruncation_h

/**
 *  Class: DynamicTruncation
 *
 *  Description:
 *  class for the dynamical stop of the KF according to the
 *  compatibility degree between the extrapolated track
 *  state and the reconstructed segment in the muon chambers
 *
 *
 *  Authors :
 *  D. Pagano & G. Bruno - UCL Louvain
 *
 *  \modified by C. Caputo, UCLouvain
 **/

#include <memory>
#include "RecoMuon/GlobalTrackingTools/interface/DirectTrackerNavigation.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "RecoMuon/GlobalTrackingTools/interface/StateSegmentMatcher.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "RecoMuon/Navigation/interface/MuonNavigableLayer.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/Navigation/interface/DirectMuonNavigation.h"
#include "DataFormats/MuonReco/interface/DYTInfo.h"
#include "RecoMuon/GlobalTrackingTools/interface/ThrParameters.h"
#include "RecoMuon/GlobalTrackingTools/interface/ChamberSegmentUtility.h"

class TransientRecHitRecord;

namespace dyt_utils {
  enum class etaRegion { eta0p8, eta1p2, eta2p0, eta2p2, eta2p4 };
};

class DynamicTruncation {
public:
  struct Config {
    Config(edm::ConsumesCollector);

    const edm::ESGetToken<CSCGeometry, MuonGeometryRecord> cscGeomToken_;
    const edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> muonRecHitBuilderToken_;
    const edm::ESGetToken<TrajectoryStateUpdator, TrackingComponentsRecord> updatorToken_;
    const edm::ESGetToken<MuonDetLayerGeometry, MuonRecoGeometryRecord> navMuonToken_;

    const edm::ESGetToken<DYTThrObject, DYTThrObjectRcd> dytThresholdsToken_;
    const edm::ESGetToken<AlignmentErrorsExtended, DTAlignmentErrorExtendedRcd> dtAlignmentErrorsToken_;
    const edm::ESGetToken<AlignmentErrorsExtended, CSCAlignmentErrorExtendedRcd> cscAlignmentErrorsToken_;
  };

  typedef TransientTrackingRecHit::ConstRecHitPointer ConstRecHitPointer;
  typedef TransientTrackingRecHit::ConstRecHitContainer ConstRecHitContainer;

  DynamicTruncation(Config const &, const edm::EventSetup &, const MuonServiceProxy &);

  ~DynamicTruncation();

  void setProd(const edm::Handle<DTRecSegment4DCollection> &DTSegProd,
               const edm::Handle<CSCSegmentCollection> &CSCSegProd) {
    getSegs->initCSU(DTSegProd, CSCSegProd);
  }

  void setSelector(int);
  void setThr(const std::vector<int> &);
  void setUpdateState(bool);
  void setUseAPE(bool);
  /*---- DyT v2-----*/
  void setThrsMap(const edm::ParameterSet &);
  void setParThrsMode(bool dytParThrsMode) { useParametrizedThr = dytParThrsMode; }
  void setRecoP(double p) { p_reco = p; }
  void setRecoEta(double eta) {
    eta_reco = eta;
    setEtaRegion();
  }

  // Return the vector with the tracker plus the selected muon hits
  TransientTrackingRecHit::ConstRecHitContainer filter(const Trajectory &);

  // Return the DYTInfo object
  reco::DYTInfo getDYTInfo() {
    dytInfo.setNStUsed(nStationsUsed);
    dytInfo.setDYTEstimators(estimatorMap);
    dytInfo.setUsedStations(usedStationMap);
    dytInfo.setIdChambers(idChamberMap);
    return dytInfo;
  }

private:
  void compatibleDets(TrajectoryStateOnSurface &, std::map<int, std::vector<DetId>> &);
  void filteringAlgo();
  void fillSegmentMaps(std::map<int, std::vector<DetId>> &,
                       std::map<int, std::vector<DTRecSegment4D>> &,
                       std::map<int, std::vector<CSCSegment>> &);
  void preliminaryFit(std::map<int, std::vector<DetId>>,
                      std::map<int, std::vector<DTRecSegment4D>>,
                      std::map<int, std::vector<CSCSegment>>);
  bool chooseLayers(int &,
                    double const &,
                    DTRecSegment4D const &,
                    TrajectoryStateOnSurface const &,
                    double const &,
                    CSCSegment const &,
                    TrajectoryStateOnSurface const &);
  void fillDYTInfos(
      int const &, bool const &, int &, double const &, double const &, DTRecSegment4D const &, CSCSegment const &);
  int stationfromDet(DetId const &);
  void update(TrajectoryStateOnSurface &, ConstRecHitPointer);
  void updateWithDThits(TrajectoryStateOnSurface &, DTRecSegment4D const &);
  void updateWithCSChits(TrajectoryStateOnSurface &, CSCSegment const &);
  void getThresholdFromDB(double &, DetId const &);
  void correctThrByPAndEta(double &);
  void getThresholdFromCFG(double &, DetId const &);
  void testDTstation(TrajectoryStateOnSurface &,
                     std::vector<DTRecSegment4D> const &,
                     double &,
                     DTRecSegment4D &,
                     TrajectoryStateOnSurface &);
  void testCSCstation(
      TrajectoryStateOnSurface &, std::vector<CSCSegment> const &, double &, CSCSegment &, TrajectoryStateOnSurface &);
  void useSegment(DTRecSegment4D const &, TrajectoryStateOnSurface const &);
  void useSegment(CSCSegment const &, TrajectoryStateOnSurface const &);
  void sort(ConstRecHitContainer &);
  void setEtaRegion();

  ConstRecHitContainer result, prelFitMeas;
  bool useAPE;
  std::vector<int> Thrs;
  int nStationsUsed;
  int DYTselector;
  edm::ESHandle<Propagator> propagator;
  edm::ESHandle<Propagator> propagatorPF;
  edm::ESHandle<Propagator> propagatorCompatibleDet;
  edm::ESHandle<GlobalTrackingGeometry> theG;
  edm::ESHandle<CSCGeometry> cscGeom;
  edm::ESHandle<TransientTrackingRecHitBuilder> theMuonRecHitBuilder;
  edm::ESHandle<TrajectoryStateUpdator> updatorHandle;
  edm::ESHandle<MuonDetLayerGeometry> navMuon;
  std::unique_ptr<DirectMuonNavigation> navigation;
  edm::ESHandle<MagneticField> magfield;
  std::map<int, double> estimatorMap;
  std::map<int, bool> usedStationMap;
  std::map<int, DetId> idChamberMap;
  TrajectoryStateOnSurface currentState;
  TrajectoryStateOnSurface prelFitState;
  reco::DYTInfo dytInfo;
  std::map<DTChamberId, GlobalError> dtApeMap;
  std::map<CSCDetId, GlobalError> cscApeMap;
  double muonPTest, muonETAest;
  const DYTThrObject *dytThresholds;
  std::unique_ptr<ChamberSegmentUtility> getSegs;
  std::unique_ptr<ThrParameters> thrManager;
  bool useDBforThr;
  bool doUpdateOfKFStates;
  /* Variables for v2 */
  double p_reco;
  double eta_reco;
  bool useParametrizedThr;
  dyt_utils::etaRegion region;
  std::map<dyt_utils::etaRegion, std::vector<double>> parameters;
};

#endif
