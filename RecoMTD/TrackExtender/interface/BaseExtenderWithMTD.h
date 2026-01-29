#ifndef RecoMTD_TrackExtender_BaseExtenderWithMTD_h
#define RecoMTD_TrackExtender_BaseExtenderWithMTD_h

#include <CLHEP/Units/GlobalPhysicalConstants.h>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "DataFormats/TrackerRecHit2D/interface/MTDTrackingRecHit.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoMTD/DetLayers/interface/MTDDetLayerGeometry.h"
#include "RecoMTD/Records/interface/MTDRecoGeometryRecord.h"
#include "RecoMTD/TransientTrackingRecHit/interface/MTDTransientTrackingRecHitBuilder.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderWithPropagator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TrackRefitter/interface/RefitDirection.h"

using namespace std;
using namespace edm;
using namespace reco;

namespace mtdtof{
  constexpr float c_cm_ns = geant_units::operators::convertMmToCm(CLHEP::c_light);  // [mm/ns] -> [cm/ns]
  constexpr float c_inv = 1.0f / c_cm_ns;

  class MTDHitMatchingInfo{
    public:
      MTDHitMatchingInfo();

      bool operator<(const MTDHitMatchingInfo& m2) const;
      float chi2(float timeWeight = 1.f) const;

      const MTDTrackingRecHit* hit;
      float estChi2;
      float timeChi2;
  };

  class TrackSegments {
    public:
      TrackSegments();

      uint32_t addSegment(float tPath, float tMom2, float sigmaMom);
      float computeTof(float mass_inv2) const;
      float computeSigmaTof(float mass_inv2);
      uint32_t size() const;
      uint32_t removeFirstSegment();
      std::pair<float, float> getSegmentPathAndMom2(uint32_t iSegment) const;

      uint32_t nSegment_ = 0;
      std::vector<float> segmentPathOvc_;
      std::vector<float> segmentMom2_;
      std::vector<float> segmentSigmaMom_;

      std::vector<float> sigmaTofs_;
  };

  struct TrackTofPidInfo {
    float tmtd;
    float tmtderror;
    float pathlength;

    float betaerror;

    float dt;
    float dterror;
    float dterror2;
    float dtchi2;

    float dt_best;
    float dterror_best;
    float dtchi2_best;

    float gammasq_pi;
    float beta_pi;
    float dt_pi;
    float sigma_dt_pi;

    float gammasq_k;
    float beta_k;
    float dt_k;
    float sigma_dt_k;

    float gammasq_p;
    float beta_p;
    float dt_p;
    float sigma_dt_p;

    float prob_pi;
    float prob_k;
    float prob_p;
  };

  enum class TofCalc { kCost = 1, kSegm = 2, kMixd = 3 };
  enum class SigmaTofCalc { kCost = 1, kSegm = 2, kMixd = 3 };

  const TrackTofPidInfo computeTrackTofPidInfo(float magp2,
                                               float length,
                                               TrackSegments trs,
                                               float t_mtd,
                                               float t_mtderr,
                                               float t_vtx,
                                               float t_vtx_err,
                                               bool addPIDError = true,
                                               TofCalc choice = TofCalc::kCost,
                                               SigmaTofCalc sigma_choice = SigmaTofCalc::kCost);

  bool getTrajectoryStateClosestToBeamLine(const Trajectory& traj,
                                           const reco::BeamSpot& bs,
                                           const Propagator* thePropagator,
                                           TrajectoryStateClosestToBeamLine& tscbl);

  bool trackPathLength(const Trajectory& traj,
                       const TrajectoryStateClosestToBeamLine& tscbl,
                       const Propagator* thePropagator,
                       float& pathlength,
                       TrackSegments& trs);

  bool trackPathLength(const Trajectory& traj,
                       const reco::BeamSpot& bs,
                       const Propagator* thePropagator,
                       float& pathlength,
                       TrackSegments& trs);

  bool cmp_for_detset(const unsigned one, const unsigned two);

  void find_hits_in_dets(const MTDTrackingDetSetVector& hits,
                         const Trajectory& traj,
                         const DetLayer* layer,
                         const TrajectoryStateOnSurface& tsos,
                         const float pmag2,
                         const float pathlength0,
                         const TrackSegments& trs0,
                         const float vtxTime,
                         const float vtxTimeError,
                         bool useVtxConstraint,
                         const reco::BeamSpot& bs,
                         const float bsTimeSpread,
                         const Propagator* prop,
                         const MeasurementEstimator* estimator,
                         std::set<MTDHitMatchingInfo>& out);
}

class BaseExtenderWithMTD {

  public:
    explicit BaseExtenderWithMTD(const ParameterSet& iConfig);
    virtual ~BaseExtenderWithMTD();

    std::unique_ptr<MeasurementEstimator> theEstimator;

    typedef typename TrackCollection::value_type TrackType;
    typedef edm::View<TrackType> InputCollection;

    void setParameters(const TransientTrackingRecHitBuilder* hitbuilder, const GlobalTrackingGeometry* gtg);
    const TransientTrackingRecHitBuilder* getHitBuilder() const;

    void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    TransientTrackingRecHit::ConstRecHitContainer tryBTLLayers(const TrajectoryStateOnSurface&,
                                                               const Trajectory& traj,
                                                               const float,
                                                               const float,
                                                               const mtdtof::TrackSegments&,
                                                               const MTDTrackingDetSetVector&,
                                                               const MTDDetLayerGeometry*,
                                                               const MagneticField* field,
                                                               const Propagator* prop,
                                                               const reco::BeamSpot& bs,
                                                               const float vtxTime,
                                                               const float vtxTimeError,
                                                               mtdtof::MTDHitMatchingInfo& bestHit) const;

    TransientTrackingRecHit::ConstRecHitContainer tryETLLayers(const TrajectoryStateOnSurface&,
                                                              const Trajectory& traj,
                                                              const float,
                                                              const float,
                                                              const mtdtof::TrackSegments&,
                                                              const MTDTrackingDetSetVector&,
                                                              const MTDDetLayerGeometry*,
                                                              const MagneticField* field,
                                                              const Propagator* prop,
                                                              const reco::BeamSpot& bs,
                                                              const float vtxTime,
                                                              const float vtxTimeError,
                                                              mtdtof::MTDHitMatchingInfo& bestHit) const;

    virtual void fillMatchingHits(const DetLayer*,
                                  const TrajectoryStateOnSurface&,
                                  const Trajectory&,
                                  const float,
                                  const float,
                                  const mtdtof::TrackSegments&,
                                  const MTDTrackingDetSetVector&,
                                  const Propagator*,
                                  const reco::BeamSpot&,
                                  const float&,
                                  const float&,
                                  TransientTrackingRecHit::ConstRecHitContainer&,
                                  mtdtof::MTDHitMatchingInfo&) const;
    
    RefitDirection::GeometricalDirection checkRecHitsOrdering(TransientTrackingRecHit::ConstRecHitContainer const& recHits) const;                          

    reco::Track buildTrack(const reco::TrackRef&,
                           const Trajectory&,
                           const Trajectory&,
                           const reco::BeamSpot&,
                           const MagneticField* field,
                           const Propagator* prop,
                           bool hasMTD,
                           float& pathLength,
                           float& tmtdOut,
                           float& sigmatmtdOut,
                           GlobalPoint& tmtdPosOut,
                           float& tofpi,
                           float& tofk,
                           float& tofp,
                           float& sigmatofpi,
                           float& sigmatofk,
                           float& sigmatofp) const;

    reco::TrackExtra buildTrackExtra(const Trajectory& trajectory) const;
    string dumpLayer(const DetLayer* layer) const; 

    const float estMaxChi2_;
    const float estMaxNSigma_;
    const float btlChi2Cut_;
    const float btlTimeChi2Cut_;
    const float etlChi2Cut_;
    const float etlTimeChi2Cut_;

    const bool useVertex_;
    const float dzCut_;
    const float bsTimeSpread_;

  private:

    const TransientTrackingRecHitBuilder* basehitbuilder_;
    const GlobalTrackingGeometry* basegtg_;

    static constexpr float trackMaxBtlEta_ = 1.5;
};

#endif
