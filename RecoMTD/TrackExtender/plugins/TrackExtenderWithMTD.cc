#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMTD/DetLayers/interface/MTDDetLayerGeometry.h"
#include "RecoMTD/Records/interface/MTDRecoGeometryRecord.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"

#include "DataFormats/TrackerRecHit2D/interface/MTDTrackingRecHit.h"

#include "RecoMTD/DetLayers/interface/MTDTrayBarrelLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include "DataFormats/ForwardDetId/interface/MTDChannelIdentifier.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include "RecoMTD/TransientTrackingRecHit/interface/MTDTransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

#include "TrackingTools/PatternTools/interface/TSCBLBuilderWithPropagator.h"

#include "RecoTracker/TransientTrackingRecHit/interface/Traj2TrackHits.h"
#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"

#include <sstream>

#include "Geometry/CommonTopologies/interface/Topology.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/Math/interface/GeantUnits.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "DataFormats/Math/interface/Rounding.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

using namespace std;
using namespace edm;
using namespace reco;

namespace {
  constexpr float c_cm_ns = geant_units::operators::convertMmToCm(CLHEP::c_light);  // [mm/ns] -> [cm/ns]
  constexpr float c_inv = 1.0f / c_cm_ns;

  class MTDHitMatchingInfo {
  public:
    MTDHitMatchingInfo() {
      hit = nullptr;
      estChi2 = std::numeric_limits<float>::max();
      timeChi2 = std::numeric_limits<float>::max();
    }

    //Operator used to sort the hits while performing the matching step at the MTD
    inline bool operator<(const MTDHitMatchingInfo& m2) const {
      //only for good matching in time use estChi2, otherwise use mostly time compatibility
      constexpr float chi2_cut = 10.f;
      constexpr float low_weight = 3.f;
      constexpr float high_weight = 8.f;
      if (timeChi2 < chi2_cut && m2.timeChi2 < chi2_cut)
        return chi2(low_weight) < m2.chi2(low_weight);
      else
        return chi2(high_weight) < m2.chi2(high_weight);
    }

    inline float chi2(float timeWeight = 1.f) const { return estChi2 + timeWeight * timeChi2; }

    const MTDTrackingRecHit* hit;
    float estChi2;
    float timeChi2;
  };

  class TrackSegments {
  public:
    TrackSegments() = default;

    inline uint32_t addSegment(float tPath, float tMom2) {
      segmentPathOvc_.emplace_back(tPath * c_inv);
      segmentMom2_.emplace_back(tMom2);
      nSegment_++;

      LogTrace("TrackExtenderWithMTD") << "addSegment # " << nSegment_ << " s = " << tPath
                                       << " p = " << std::sqrt(tMom2);

      return nSegment_;
    }

    inline float computeTof(float mass_inv2) const {
      float tof(0.f);
      for (uint32_t iSeg = 0; iSeg < nSegment_; iSeg++) {
        float gammasq = 1.f + segmentMom2_[iSeg] * mass_inv2;
        float beta = std::sqrt(1.f - 1.f / gammasq);
        tof += segmentPathOvc_[iSeg] / beta;

        LogTrace("TrackExtenderWithMTD") << " TOF Segment # " << iSeg + 1 << " p = " << std::sqrt(segmentMom2_[iSeg])
                                         << " tof = " << tof;
      }

      return tof;
    }

    inline uint32_t size() const { return nSegment_; }

    inline uint32_t removeFirstSegment() {
      if (nSegment_ > 0) {
        segmentPathOvc_.erase(segmentPathOvc_.begin());
        segmentMom2_.erase(segmentMom2_.begin());
        nSegment_--;
      }
      return nSegment_;
    }

    inline std::pair<float, float> getSegmentPathAndMom2(uint32_t iSegment) const {
      if (iSegment >= nSegment_) {
        throw cms::Exception("TrackExtenderWithMTD") << "Requesting non existing track segment #" << iSegment;
      }
      return std::make_pair(segmentPathOvc_[iSegment], segmentMom2_[iSegment]);
    }

    uint32_t nSegment_ = 0;
    std::vector<float> segmentPathOvc_;
    std::vector<float> segmentMom2_;
  };

  struct TrackTofPidInfo {
    float tmtd;
    float tmtderror;
    float pathlength;

    float betaerror;

    float dt;
    float dterror;
    float dtchi2;

    float dt_best;
    float dterror_best;
    float dtchi2_best;

    float gammasq_pi;
    float beta_pi;
    float dt_pi;

    float gammasq_k;
    float beta_k;
    float dt_k;

    float gammasq_p;
    float beta_p;
    float dt_p;

    float prob_pi;
    float prob_k;
    float prob_p;
  };

  enum class TofCalc { kCost = 1, kSegm = 2, kMixd = 3 };

  const TrackTofPidInfo computeTrackTofPidInfo(float magp2,
                                               float length,
                                               TrackSegments trs,
                                               float t_mtd,
                                               float t_mtderr,
                                               float t_vtx,
                                               float t_vtx_err,
                                               bool addPIDError = true,
                                               TofCalc choice = TofCalc::kCost) {
    constexpr float m_pi = 0.13957018f;
    constexpr float m_pi_inv2 = 1.0f / m_pi / m_pi;
    constexpr float m_k = 0.493677f;
    constexpr float m_k_inv2 = 1.0f / m_k / m_k;
    constexpr float m_p = 0.9382720813f;
    constexpr float m_p_inv2 = 1.0f / m_p / m_p;

    TrackTofPidInfo tofpid;

    tofpid.tmtd = t_mtd;
    tofpid.tmtderror = t_mtderr;
    tofpid.pathlength = length;

    auto deltat = [&](const float mass_inv2, const float betatmp) {
      float res(1.f);
      switch (choice) {
        case TofCalc::kCost:
          res = tofpid.pathlength / betatmp * c_inv;
          break;
        case TofCalc::kSegm:
          res = trs.computeTof(mass_inv2);
          break;
        case TofCalc::kMixd:
          res = trs.computeTof(mass_inv2) + tofpid.pathlength / betatmp * c_inv;
          break;
      }
      return res;
    };

    tofpid.gammasq_pi = 1.f + magp2 * m_pi_inv2;
    tofpid.beta_pi = std::sqrt(1.f - 1.f / tofpid.gammasq_pi);
    tofpid.dt_pi = deltat(m_pi_inv2, tofpid.beta_pi);

    tofpid.gammasq_k = 1.f + magp2 * m_k_inv2;
    tofpid.beta_k = std::sqrt(1.f - 1.f / tofpid.gammasq_k);
    tofpid.dt_k = deltat(m_k_inv2, tofpid.beta_k);

    tofpid.gammasq_p = 1.f + magp2 * m_p_inv2;
    tofpid.beta_p = std::sqrt(1.f - 1.f / tofpid.gammasq_p);
    tofpid.dt_p = deltat(m_p_inv2, tofpid.beta_p);

    tofpid.dt = tofpid.tmtd - tofpid.dt_pi - t_vtx;  //assume by default the pi hypothesis
    tofpid.dterror = sqrt(tofpid.tmtderror * tofpid.tmtderror + t_vtx_err * t_vtx_err);
    tofpid.betaerror = 0.f;
    if (addPIDError) {
      tofpid.dterror =
          sqrt(tofpid.dterror * tofpid.dterror + (tofpid.dt_p - tofpid.dt_pi) * (tofpid.dt_p - tofpid.dt_pi));
      tofpid.betaerror = tofpid.beta_p - tofpid.beta_pi;
    }

    tofpid.dtchi2 = (tofpid.dt * tofpid.dt) / (tofpid.dterror * tofpid.dterror);

    tofpid.dt_best = tofpid.dt;
    tofpid.dterror_best = tofpid.dterror;
    tofpid.dtchi2_best = tofpid.dtchi2;

    tofpid.prob_pi = -1.f;
    tofpid.prob_k = -1.f;
    tofpid.prob_p = -1.f;

    if (!addPIDError) {
      //*TODO* deal with heavier nucleons and/or BSM case here?
      float chi2_pi = tofpid.dtchi2;
      float chi2_k =
          (tofpid.tmtd - tofpid.dt_k - t_vtx) * (tofpid.tmtd - tofpid.dt_k - t_vtx) / (tofpid.dterror * tofpid.dterror);
      float chi2_p =
          (tofpid.tmtd - tofpid.dt_p - t_vtx) * (tofpid.tmtd - tofpid.dt_p - t_vtx) / (tofpid.dterror * tofpid.dterror);

      float rawprob_pi = exp(-0.5f * chi2_pi);
      float rawprob_k = exp(-0.5f * chi2_k);
      float rawprob_p = exp(-0.5f * chi2_p);
      float normprob = 1.f / (rawprob_pi + rawprob_k + rawprob_p);

      tofpid.prob_pi = rawprob_pi * normprob;
      tofpid.prob_k = rawprob_k * normprob;
      tofpid.prob_p = rawprob_p * normprob;

      float prob_heavy = 1.f - tofpid.prob_pi;
      constexpr float heavy_threshold = 0.75f;

      if (prob_heavy > heavy_threshold) {
        if (chi2_k < chi2_p) {
          tofpid.dt_best = (tofpid.tmtd - tofpid.dt_k - t_vtx);
          tofpid.dtchi2_best = chi2_k;
        } else {
          tofpid.dt_best = (tofpid.tmtd - tofpid.dt_p - t_vtx);
          tofpid.dtchi2_best = chi2_p;
        }
      }
    }
    return tofpid;
  }

  bool getTrajectoryStateClosestToBeamLine(const Trajectory& traj,
                                           const reco::BeamSpot& bs,
                                           const Propagator* thePropagator,
                                           TrajectoryStateClosestToBeamLine& tscbl) {
    // get the state closest to the beamline
    TrajectoryStateOnSurface stateForProjectionToBeamLineOnSurface =
        traj.closestMeasurement(GlobalPoint(bs.x0(), bs.y0(), bs.z0())).updatedState();

    if (!stateForProjectionToBeamLineOnSurface.isValid()) {
      edm::LogError("CannotPropagateToBeamLine") << "the state on the closest measurement isnot valid. skipping track.";
      return false;
    }

    const FreeTrajectoryState& stateForProjectionToBeamLine = *stateForProjectionToBeamLineOnSurface.freeState();

    TSCBLBuilderWithPropagator tscblBuilder(*thePropagator);
    tscbl = tscblBuilder(stateForProjectionToBeamLine, bs);

    return tscbl.isValid();
  }

  bool trackPathLength(const Trajectory& traj,
                       const TrajectoryStateClosestToBeamLine& tscbl,
                       const Propagator* thePropagator,
                       float& pathlength,
                       TrackSegments& trs) {
    pathlength = 0.f;

    bool validpropagation = true;
    float oldp = traj.measurements().begin()->updatedState().globalMomentum().mag();
    float pathlength1 = 0.f;
    float pathlength2 = 0.f;

    //add pathlength layer by layer
    for (auto it = traj.measurements().begin(); it != traj.measurements().end() - 1; ++it) {
      const auto& propresult = thePropagator->propagateWithPath(it->updatedState(), (it + 1)->updatedState().surface());
      float layerpathlength = std::abs(propresult.second);
      if (layerpathlength == 0.f) {
        validpropagation = false;
      }
      pathlength1 += layerpathlength;
      trs.addSegment(layerpathlength, (it + 1)->updatedState().globalMomentum().mag2());
      LogTrace("TrackExtenderWithMTD") << "TSOS " << std::fixed << std::setw(4) << trs.size() << " R_i " << std::fixed
                                       << std::setw(14) << it->updatedState().globalPosition().perp() << " z_i "
                                       << std::fixed << std::setw(14) << it->updatedState().globalPosition().z()
                                       << " R_e " << std::fixed << std::setw(14)
                                       << (it + 1)->updatedState().globalPosition().perp() << " z_e " << std::fixed
                                       << std::setw(14) << (it + 1)->updatedState().globalPosition().z() << " p "
                                       << std::fixed << std::setw(14) << (it + 1)->updatedState().globalMomentum().mag()
                                       << " dp " << std::fixed << std::setw(14)
                                       << (it + 1)->updatedState().globalMomentum().mag() - oldp;
      oldp = (it + 1)->updatedState().globalMomentum().mag();
    }

    //add distance from bs to first measurement
    auto const& tscblPCA = tscbl.trackStateAtPCA();
    auto const& aSurface = traj.direction() == alongMomentum ? traj.firstMeasurement().updatedState().surface()
                                                             : traj.lastMeasurement().updatedState().surface();
    pathlength2 = thePropagator->propagateWithPath(tscblPCA, aSurface).second;
    if (pathlength2 == 0.f) {
      validpropagation = false;
    }
    pathlength = pathlength1 + pathlength2;
    trs.addSegment(pathlength2, tscblPCA.momentum().mag2());
    LogTrace("TrackExtenderWithMTD") << "TSOS " << std::fixed << std::setw(4) << trs.size() << " R_e " << std::fixed
                                     << std::setw(14) << tscblPCA.position().perp() << " z_e " << std::fixed
                                     << std::setw(14) << tscblPCA.position().z() << " p " << std::fixed << std::setw(14)
                                     << tscblPCA.momentum().mag() << " dp " << std::fixed << std::setw(14)
                                     << tscblPCA.momentum().mag() - oldp;
    return validpropagation;
  }

  bool trackPathLength(const Trajectory& traj,
                       const reco::BeamSpot& bs,
                       const Propagator* thePropagator,
                       float& pathlength,
                       TrackSegments& trs) {
    pathlength = 0.f;

    TrajectoryStateClosestToBeamLine tscbl;
    bool tscbl_status = getTrajectoryStateClosestToBeamLine(traj, bs, thePropagator, tscbl);

    if (!tscbl_status)
      return false;

    return trackPathLength(traj, tscbl, thePropagator, pathlength, trs);
  }

}  // namespace

template <class TrackCollection>
class TrackExtenderWithMTDT : public edm::stream::EDProducer<> {
public:
  typedef typename TrackCollection::value_type TrackType;
  typedef edm::View<TrackType> InputCollection;

  TrackExtenderWithMTDT(const ParameterSet& pset);

  template <class H, class T>
  void fillValueMap(edm::Event& iEvent, const H& handle, const std::vector<T>& vec, const edm::EDPutToken& token) const;

  void produce(edm::Event& ev, const edm::EventSetup& es) final;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  TransientTrackingRecHit::ConstRecHitContainer tryBTLLayers(const TrajectoryStateOnSurface&,
                                                             const Trajectory& traj,
                                                             const float,
                                                             const float,
                                                             const TrackSegments&,
                                                             const MTDTrackingDetSetVector&,
                                                             const MTDDetLayerGeometry*,
                                                             const MagneticField* field,
                                                             const Propagator* prop,
                                                             const reco::BeamSpot& bs,
                                                             const float vtxTime,
                                                             const bool matchVertex,
                                                             MTDHitMatchingInfo& bestHit) const;

  TransientTrackingRecHit::ConstRecHitContainer tryETLLayers(const TrajectoryStateOnSurface&,
                                                             const Trajectory& traj,
                                                             const float,
                                                             const float,
                                                             const TrackSegments&,
                                                             const MTDTrackingDetSetVector&,
                                                             const MTDDetLayerGeometry*,
                                                             const MagneticField* field,
                                                             const Propagator* prop,
                                                             const reco::BeamSpot& bs,
                                                             const float vtxTime,
                                                             const bool matchVertex,
                                                             MTDHitMatchingInfo& bestHit) const;

  void fillMatchingHits(const DetLayer*,
                        const TrajectoryStateOnSurface&,
                        const Trajectory&,
                        const float,
                        const float,
                        const TrackSegments&,
                        const MTDTrackingDetSetVector&,
                        const Propagator*,
                        const reco::BeamSpot&,
                        const float&,
                        const bool,
                        TransientTrackingRecHit::ConstRecHitContainer&,
                        MTDHitMatchingInfo&) const;

  RefitDirection::GeometricalDirection checkRecHitsOrdering(
      TransientTrackingRecHit::ConstRecHitContainer const& recHits) const {
    if (!recHits.empty()) {
      GlobalPoint first = gtg_->idToDet(recHits.front()->geographicalId())->position();
      GlobalPoint last = gtg_->idToDet(recHits.back()->geographicalId())->position();

      // maybe perp2?
      auto rFirst = first.mag2();
      auto rLast = last.mag2();
      if (rFirst < rLast)
        return RefitDirection::insideOut;
      if (rFirst > rLast)
        return RefitDirection::outsideIn;
    }
    LogDebug("TrackExtenderWithMTD") << "Impossible to determine the rechits order" << endl;
    return RefitDirection::undetermined;
  }

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
                         float& tofpi,
                         float& tofk,
                         float& tofp) const;
  reco::TrackExtra buildTrackExtra(const Trajectory& trajectory) const;

  string dumpLayer(const DetLayer* layer) const;

private:
  edm::EDPutToken btlMatchChi2Token_;
  edm::EDPutToken etlMatchChi2Token_;
  edm::EDPutToken btlMatchTimeChi2Token_;
  edm::EDPutToken etlMatchTimeChi2Token_;
  edm::EDPutToken npixBarrelToken_;
  edm::EDPutToken npixEndcapToken_;
  edm::EDPutToken pOrigTrkToken_;
  edm::EDPutToken betaOrigTrkToken_;
  edm::EDPutToken t0OrigTrkToken_;
  edm::EDPutToken sigmat0OrigTrkToken_;
  edm::EDPutToken pathLengthOrigTrkToken_;
  edm::EDPutToken tmtdOrigTrkToken_;
  edm::EDPutToken sigmatmtdOrigTrkToken_;
  edm::EDPutToken tofpiOrigTrkToken_;
  edm::EDPutToken tofkOrigTrkToken_;
  edm::EDPutToken tofpOrigTrkToken_;
  edm::EDPutToken assocOrigTrkToken_;

  edm::EDGetTokenT<InputCollection> tracksToken_;
  edm::EDGetTokenT<TrajTrackAssociationCollection> trajTrackAToken_;
  edm::EDGetTokenT<MTDTrackingDetSetVector> hitsToken_;
  edm::EDGetTokenT<reco::BeamSpot> bsToken_;
  edm::EDGetTokenT<GlobalPoint> genVtxPositionToken_;
  edm::EDGetTokenT<float> genVtxTimeToken_;
  edm::EDGetTokenT<VertexCollection> vtxToken_;

  const bool updateTraj_, updateExtra_, updatePattern_;
  const std::string mtdRecHitBuilder_, propagator_, transientTrackBuilder_;
  std::unique_ptr<MeasurementEstimator> theEstimator;
  std::unique_ptr<TrackTransformer> theTransformer;
  edm::ESHandle<TransientTrackBuilder> builder_;
  edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> builderToken_;
  edm::ESHandle<TransientTrackingRecHitBuilder> hitbuilder_;
  edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> hitbuilderToken_;
  edm::ESHandle<GlobalTrackingGeometry> gtg_;
  edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> gtgToken_;

  edm::ESGetToken<MTDDetLayerGeometry, MTDRecoGeometryRecord> dlgeoToken_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magfldToken_;
  edm::ESGetToken<Propagator, TrackingComponentsRecord> propToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> ttopoToken_;

  const float estMaxChi2_;
  const float estMaxNSigma_;
  const float btlChi2Cut_;
  const float btlTimeChi2Cut_;
  const float etlChi2Cut_;
  const float etlTimeChi2Cut_;

  const bool useVertex_;
  const bool useSimVertex_;
  const float dzCut_;
  const float bsTimeSpread_;
};

template <class TrackCollection>
TrackExtenderWithMTDT<TrackCollection>::TrackExtenderWithMTDT(const ParameterSet& iConfig)
    : tracksToken_(consumes<InputCollection>(iConfig.getParameter<edm::InputTag>("tracksSrc"))),
      trajTrackAToken_(consumes<TrajTrackAssociationCollection>(iConfig.getParameter<edm::InputTag>("trjtrkAssSrc"))),
      hitsToken_(consumes<MTDTrackingDetSetVector>(iConfig.getParameter<edm::InputTag>("hitsSrc"))),
      bsToken_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpotSrc"))),
      updateTraj_(iConfig.getParameter<bool>("updateTrackTrajectory")),
      updateExtra_(iConfig.getParameter<bool>("updateTrackExtra")),
      updatePattern_(iConfig.getParameter<bool>("updateTrackHitPattern")),
      mtdRecHitBuilder_(iConfig.getParameter<std::string>("MTDRecHitBuilder")),
      propagator_(iConfig.getParameter<std::string>("Propagator")),
      transientTrackBuilder_(iConfig.getParameter<std::string>("TransientTrackBuilder")),
      estMaxChi2_(iConfig.getParameter<double>("estimatorMaxChi2")),
      estMaxNSigma_(iConfig.getParameter<double>("estimatorMaxNSigma")),
      btlChi2Cut_(iConfig.getParameter<double>("btlChi2Cut")),
      btlTimeChi2Cut_(iConfig.getParameter<double>("btlTimeChi2Cut")),
      etlChi2Cut_(iConfig.getParameter<double>("etlChi2Cut")),
      etlTimeChi2Cut_(iConfig.getParameter<double>("etlTimeChi2Cut")),
      useVertex_(iConfig.getParameter<bool>("useVertex")),
      useSimVertex_(iConfig.getParameter<bool>("useSimVertex")),
      dzCut_(iConfig.getParameter<double>("dZCut")),
      bsTimeSpread_(iConfig.getParameter<double>("bsTimeSpread")) {
  if (useVertex_) {
    if (useSimVertex_) {
      genVtxPositionToken_ = consumes<GlobalPoint>(iConfig.getParameter<edm::InputTag>("genVtxPositionSrc"));
      genVtxTimeToken_ = consumes<float>(iConfig.getParameter<edm::InputTag>("genVtxTimeSrc"));
    } else
      vtxToken_ = consumes<VertexCollection>(iConfig.getParameter<edm::InputTag>("vtxSrc"));
  }

  theEstimator = std::make_unique<Chi2MeasurementEstimator>(estMaxChi2_, estMaxNSigma_);
  theTransformer = std::make_unique<TrackTransformer>(iConfig.getParameterSet("TrackTransformer"), consumesCollector());

  btlMatchChi2Token_ = produces<edm::ValueMap<float>>("btlMatchChi2");
  etlMatchChi2Token_ = produces<edm::ValueMap<float>>("etlMatchChi2");
  btlMatchTimeChi2Token_ = produces<edm::ValueMap<float>>("btlMatchTimeChi2");
  etlMatchTimeChi2Token_ = produces<edm::ValueMap<float>>("etlMatchTimeChi2");
  npixBarrelToken_ = produces<edm::ValueMap<int>>("npixBarrel");
  npixEndcapToken_ = produces<edm::ValueMap<int>>("npixEndcap");
  pOrigTrkToken_ = produces<edm::ValueMap<float>>("generalTrackp");
  betaOrigTrkToken_ = produces<edm::ValueMap<float>>("generalTrackBeta");
  t0OrigTrkToken_ = produces<edm::ValueMap<float>>("generalTrackt0");
  sigmat0OrigTrkToken_ = produces<edm::ValueMap<float>>("generalTracksigmat0");
  pathLengthOrigTrkToken_ = produces<edm::ValueMap<float>>("generalTrackPathLength");
  tmtdOrigTrkToken_ = produces<edm::ValueMap<float>>("generalTracktmtd");
  sigmatmtdOrigTrkToken_ = produces<edm::ValueMap<float>>("generalTracksigmatmtd");
  tofpiOrigTrkToken_ = produces<edm::ValueMap<float>>("generalTrackTofPi");
  tofkOrigTrkToken_ = produces<edm::ValueMap<float>>("generalTrackTofK");
  tofpOrigTrkToken_ = produces<edm::ValueMap<float>>("generalTrackTofP");
  assocOrigTrkToken_ = produces<edm::ValueMap<int>>("generalTrackassoc");

  builderToken_ = esConsumes<TransientTrackBuilder, TransientTrackRecord>(edm::ESInputTag("", transientTrackBuilder_));
  hitbuilderToken_ =
      esConsumes<TransientTrackingRecHitBuilder, TransientRecHitRecord>(edm::ESInputTag("", mtdRecHitBuilder_));
  gtgToken_ = esConsumes<GlobalTrackingGeometry, GlobalTrackingGeometryRecord>();
  dlgeoToken_ = esConsumes<MTDDetLayerGeometry, MTDRecoGeometryRecord>();
  magfldToken_ = esConsumes<MagneticField, IdealMagneticFieldRecord>();
  propToken_ = esConsumes<Propagator, TrackingComponentsRecord>(edm::ESInputTag("", propagator_));
  ttopoToken_ = esConsumes<TrackerTopology, TrackerTopologyRcd>();

  produces<edm::OwnVector<TrackingRecHit>>();
  produces<reco::TrackExtraCollection>();
  produces<TrackCollection>();
}

template <class TrackCollection>
void TrackExtenderWithMTDT<TrackCollection>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc, transDesc;
  desc.add<edm::InputTag>("tracksSrc", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("trjtrkAssSrc", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("hitsSrc", edm::InputTag("mtdTrackingRecHits"));
  desc.add<edm::InputTag>("beamSpotSrc", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("genVtxPositionSrc", edm::InputTag("genParticles:xyz0"));
  desc.add<edm::InputTag>("genVtxTimeSrc", edm::InputTag("genParticles:t0"));
  desc.add<edm::InputTag>("vtxSrc", edm::InputTag("offlinePrimaryVertices4D"));
  desc.add<bool>("updateTrackTrajectory", true);
  desc.add<bool>("updateTrackExtra", true);
  desc.add<bool>("updateTrackHitPattern", true);
  desc.add<std::string>("TransientTrackBuilder", "TransientTrackBuilder");
  desc.add<std::string>("MTDRecHitBuilder", "MTDRecHitBuilder");
  desc.add<std::string>("Propagator", "PropagatorWithMaterialForMTD");
  TrackTransformer::fillPSetDescription(transDesc,
                                        false,
                                        "KFFitterForRefitInsideOut",
                                        "KFSmootherForRefitInsideOut",
                                        "PropagatorWithMaterialForMTD",
                                        "alongMomentum",
                                        true,
                                        "WithTrackAngle",
                                        "MuonRecHitBuilder",
                                        "MTDRecHitBuilder");
  desc.add<edm::ParameterSetDescription>("TrackTransformer", transDesc);
  desc.add<double>("estimatorMaxChi2", 500.);
  desc.add<double>("estimatorMaxNSigma", 10.);
  desc.add<double>("btlChi2Cut", 50.);
  desc.add<double>("btlTimeChi2Cut", 10.);
  desc.add<double>("etlChi2Cut", 50.);
  desc.add<double>("etlTimeChi2Cut", 10.);
  desc.add<bool>("useVertex", false);
  desc.add<bool>("useSimVertex", false);
  desc.add<double>("dZCut", 0.1);
  desc.add<double>("bsTimeSpread", 0.2);
  descriptions.add("trackExtenderWithMTDBase", desc);
}

template <class TrackCollection>
template <class H, class T>
void TrackExtenderWithMTDT<TrackCollection>::fillValueMap(edm::Event& iEvent,
                                                          const H& handle,
                                                          const std::vector<T>& vec,
                                                          const edm::EDPutToken& token) const {
  auto out = std::make_unique<edm::ValueMap<T>>();
  typename edm::ValueMap<T>::Filler filler(*out);
  filler.insert(handle, vec.begin(), vec.end());
  filler.fill();
  iEvent.put(token, std::move(out));
}

template <class TrackCollection>
void TrackExtenderWithMTDT<TrackCollection>::produce(edm::Event& ev, const edm::EventSetup& es) {
  //this produces pieces of the track extra
  Traj2TrackHits t2t;

  theTransformer->setServices(es);

  TrackingRecHitRefProd hitsRefProd = ev.getRefBeforePut<TrackingRecHitCollection>();
  reco::TrackExtraRefProd extrasRefProd = ev.getRefBeforePut<reco::TrackExtraCollection>();

  gtg_ = es.getHandle(gtgToken_);

  auto geo = es.getTransientHandle(dlgeoToken_);

  auto magfield = es.getTransientHandle(magfldToken_);

  builder_ = es.getHandle(builderToken_);
  hitbuilder_ = es.getHandle(hitbuilderToken_);

  auto propH = es.getTransientHandle(propToken_);
  const Propagator* prop = propH.product();

  auto httopo = es.getTransientHandle(ttopoToken_);
  const TrackerTopology& ttopo = *httopo;

  auto output = std::make_unique<TrackCollection>();
  auto extras = std::make_unique<reco::TrackExtraCollection>();
  auto outhits = std::make_unique<edm::OwnVector<TrackingRecHit>>();

  std::vector<float> btlMatchChi2;
  std::vector<float> etlMatchChi2;
  std::vector<float> btlMatchTimeChi2;
  std::vector<float> etlMatchTimeChi2;
  std::vector<int> npixBarrel;
  std::vector<int> npixEndcap;
  std::vector<float> pOrigTrkRaw;
  std::vector<float> betaOrigTrkRaw;
  std::vector<float> t0OrigTrkRaw;
  std::vector<float> sigmat0OrigTrkRaw;
  std::vector<float> pathLengthsOrigTrkRaw;
  std::vector<float> tmtdOrigTrkRaw;
  std::vector<float> sigmatmtdOrigTrkRaw;
  std::vector<float> tofpiOrigTrkRaw;
  std::vector<float> tofkOrigTrkRaw;
  std::vector<float> tofpOrigTrkRaw;
  std::vector<int> assocOrigTrkRaw;

  auto const tracksH = ev.getHandle(tracksToken_);

  const auto& trjtrks = ev.get(trajTrackAToken_);

  //MTD hits DetSet
  const auto& hits = ev.get(hitsToken_);

  //beam spot
  const auto& bs = ev.get(bsToken_);

  const Vertex* pv = nullptr;
  if (useVertex_ && !useSimVertex_) {
    auto const& vtxs = ev.get(vtxToken_);
    if (!vtxs.empty())
      pv = &vtxs[0];
  }

  std::unique_ptr<math::XYZTLorentzVectorF> genPV(nullptr);
  if (useVertex_ && useSimVertex_) {
    const auto& genVtxPosition = ev.get(genVtxPositionToken_);
    const auto& genVtxTime = ev.get(genVtxTimeToken_);
    genPV = std::make_unique<math::XYZTLorentzVectorF>(
        genVtxPosition.x(), genVtxPosition.y(), genVtxPosition.z(), genVtxTime);
  }

  float vtxTime = 0.f;
  if (useVertex_) {
    if (useSimVertex_ && genPV) {
      vtxTime = genPV->t();
    } else if (pv)
      vtxTime = pv->t();  //already in ns
  }

  std::vector<unsigned> track_indices;
  unsigned itrack = 0;

  for (const auto& trjtrk : trjtrks) {
    const Trajectory& trajs = *trjtrk.key;
    const reco::TrackRef& track = trjtrk.val;

    LogTrace("TrackExtenderWithMTD") << "TrackExtenderWithMTD: extrapolating track " << itrack
                                     << " p/pT = " << track->p() << " " << track->pt() << " eta = " << track->eta();

    float trackVtxTime = 0.f;
    if (useVertex_) {
      float dz;
      if (useSimVertex_)
        dz = std::abs(track->dz(math::XYZPoint(*genPV)));
      else
        dz = std::abs(track->dz(pv->position()));

      if (dz < dzCut_)
        trackVtxTime = vtxTime;
    }

    reco::TransientTrack ttrack(track, magfield.product(), gtg_);
    auto thits = theTransformer->getTransientRecHits(ttrack);
    TransientTrackingRecHit::ConstRecHitContainer mtdthits;
    MTDHitMatchingInfo mBTL, mETL;

    if (trajs.isValid()) {
      // get the outermost trajectory point on the track
      TrajectoryStateOnSurface tsos = builder_->build(track).outermostMeasurementState();
      TrajectoryStateClosestToBeamLine tscbl;
      bool tscbl_status = getTrajectoryStateClosestToBeamLine(trajs, bs, prop, tscbl);

      if (tscbl_status) {
        float pmag2 = tscbl.trackStateAtPCA().momentum().mag2();
        float pathlength0;
        TrackSegments trs0;
        trackPathLength(trajs, tscbl, prop, pathlength0, trs0);

        const auto& btlhits = tryBTLLayers(tsos,
                                           trajs,
                                           pmag2,
                                           pathlength0,
                                           trs0,
                                           hits,
                                           geo.product(),
                                           magfield.product(),
                                           prop,
                                           bs,
                                           trackVtxTime,
                                           trackVtxTime != 0.f,
                                           mBTL);
        mtdthits.insert(mtdthits.end(), btlhits.begin(), btlhits.end());

        // in the future this should include an intermediate refit before propagating to the ETL
        // for now it is ok
        const auto& etlhits = tryETLLayers(tsos,
                                           trajs,
                                           pmag2,
                                           pathlength0,
                                           trs0,
                                           hits,
                                           geo.product(),
                                           magfield.product(),
                                           prop,
                                           bs,
                                           trackVtxTime,
                                           trackVtxTime != 0.f,
                                           mETL);
        mtdthits.insert(mtdthits.end(), etlhits.begin(), etlhits.end());
      }
    }

    auto ordering = checkRecHitsOrdering(thits);
    if (ordering == RefitDirection::insideOut) {
      thits.insert(thits.end(), mtdthits.begin(), mtdthits.end());
    } else {
      std::reverse(mtdthits.begin(), mtdthits.end());
      mtdthits.insert(mtdthits.end(), thits.begin(), thits.end());
      thits.swap(mtdthits);
    }

    const auto& trajwithmtd =
        mtdthits.empty() ? std::vector<Trajectory>(1, trajs) : theTransformer->transform(ttrack, thits);
    float pMap = 0.f, betaMap = 0.f, t0Map = 0.f, sigmat0Map = -1.f, pathLengthMap = -1.f, tmtdMap = 0.f,
          sigmatmtdMap = -1.f, tofpiMap = 0.f, tofkMap = 0.f, tofpMap = 0.f;
    int iMap = -1;

    for (const auto& trj : trajwithmtd) {
      const auto& thetrj = (updateTraj_ ? trj : trajs);
      float pathLength = 0.f, tmtd = 0.f, sigmatmtd = -1.f, tofpi = 0.f, tofk = 0.f, tofp = 0.f;
      LogTrace("TrackExtenderWithMTD") << "TrackExtenderWithMTD: refit track " << itrack << " p/pT = " << track->p()
                                       << " " << track->pt() << " eta = " << track->eta();
      reco::Track result = buildTrack(track,
                                      thetrj,
                                      trj,
                                      bs,
                                      magfield.product(),
                                      prop,
                                      !trajwithmtd.empty() && !mtdthits.empty(),
                                      pathLength,
                                      tmtd,
                                      sigmatmtd,
                                      tofpi,
                                      tofk,
                                      tofp);
      if (result.ndof() >= 0) {
        /// setup the track extras
        reco::TrackExtra::TrajParams trajParams;
        reco::TrackExtra::Chi2sFive chi2s;
        size_t hitsstart = outhits->size();
        if (updatePattern_) {
          t2t(trj, *outhits, trajParams, chi2s);  // this fills the output hit collection
        } else {
          t2t(thetrj, *outhits, trajParams, chi2s);
        }
        size_t hitsend = outhits->size();
        extras->push_back(buildTrackExtra(trj));  // always push back the fully built extra, update by setting in track
        extras->back().setHits(hitsRefProd, hitsstart, hitsend - hitsstart);
        extras->back().setTrajParams(trajParams, chi2s);
        //create the track
        output->push_back(result);
        btlMatchChi2.push_back(mBTL.hit ? mBTL.estChi2 : -1.f);
        etlMatchChi2.push_back(mETL.hit ? mETL.estChi2 : -1.f);
        btlMatchTimeChi2.push_back(mBTL.hit ? mBTL.timeChi2 : -1.f);
        etlMatchTimeChi2.push_back(mETL.hit ? mETL.timeChi2 : -1.f);
        pathLengthMap = pathLength;
        tmtdMap = tmtd;
        sigmatmtdMap = sigmatmtd;
        auto& backtrack = output->back();
        iMap = output->size() - 1;
        pMap = backtrack.p();
        betaMap = backtrack.beta();
        t0Map = backtrack.t0();
        sigmat0Map = std::copysign(std::sqrt(std::abs(backtrack.covt0t0())), backtrack.covt0t0());
        tofpiMap = tofpi;
        tofkMap = tofk;
        tofpMap = tofp;
        reco::TrackExtraRef extraRef(extrasRefProd, extras->size() - 1);
        backtrack.setExtra((updateExtra_ ? extraRef : track->extra()));
        for (unsigned ihit = hitsstart; ihit < hitsend; ++ihit) {
          backtrack.appendHitPattern((*outhits)[ihit], ttopo);
        }
        npixBarrel.push_back(backtrack.hitPattern().numberOfValidPixelBarrelHits());
        npixEndcap.push_back(backtrack.hitPattern().numberOfValidPixelEndcapHits());
        LogTrace("TrackExtenderWithMTD") << "TrackExtenderWithMTD: tmtd " << tmtdMap << " +/- " << sigmatmtdMap
                                         << " t0 " << t0Map << " +/- " << sigmat0Map << " tof pi/K/p " << tofpiMap
                                         << " " << tofkMap << " " << tofpMap;
      } else {
        LogTrace("TrackExtenderWithMTD") << "Error in the MTD track refitting. This should not happen";
      }
    }

    pOrigTrkRaw.push_back(pMap);
    betaOrigTrkRaw.push_back(betaMap);
    t0OrigTrkRaw.push_back(t0Map);
    sigmat0OrigTrkRaw.push_back(sigmat0Map);
    pathLengthsOrigTrkRaw.push_back(pathLengthMap);
    tmtdOrigTrkRaw.push_back(tmtdMap);
    sigmatmtdOrigTrkRaw.push_back(sigmatmtdMap);
    tofpiOrigTrkRaw.push_back(tofpiMap);
    tofkOrigTrkRaw.push_back(tofkMap);
    tofpOrigTrkRaw.push_back(tofpMap);
    assocOrigTrkRaw.push_back(iMap);

    if (iMap == -1) {
      btlMatchChi2.push_back(-1.f);
      etlMatchChi2.push_back(-1.f);
      btlMatchTimeChi2.push_back(-1.f);
      etlMatchTimeChi2.push_back(-1.f);
      npixBarrel.push_back(-1.f);
      npixEndcap.push_back(-1.f);
    }

    ++itrack;
  }

  auto outTrksHandle = ev.put(std::move(output));
  ev.put(std::move(extras));
  ev.put(std::move(outhits));

  fillValueMap(ev, tracksH, btlMatchChi2, btlMatchChi2Token_);
  fillValueMap(ev, tracksH, etlMatchChi2, etlMatchChi2Token_);
  fillValueMap(ev, tracksH, btlMatchTimeChi2, btlMatchTimeChi2Token_);
  fillValueMap(ev, tracksH, etlMatchTimeChi2, etlMatchTimeChi2Token_);
  fillValueMap(ev, tracksH, npixBarrel, npixBarrelToken_);
  fillValueMap(ev, tracksH, npixEndcap, npixEndcapToken_);
  fillValueMap(ev, tracksH, pOrigTrkRaw, pOrigTrkToken_);
  fillValueMap(ev, tracksH, betaOrigTrkRaw, betaOrigTrkToken_);
  fillValueMap(ev, tracksH, t0OrigTrkRaw, t0OrigTrkToken_);
  fillValueMap(ev, tracksH, sigmat0OrigTrkRaw, sigmat0OrigTrkToken_);
  fillValueMap(ev, tracksH, pathLengthsOrigTrkRaw, pathLengthOrigTrkToken_);
  fillValueMap(ev, tracksH, tmtdOrigTrkRaw, tmtdOrigTrkToken_);
  fillValueMap(ev, tracksH, sigmatmtdOrigTrkRaw, sigmatmtdOrigTrkToken_);
  fillValueMap(ev, tracksH, tofpiOrigTrkRaw, tofpiOrigTrkToken_);
  fillValueMap(ev, tracksH, tofkOrigTrkRaw, tofkOrigTrkToken_);
  fillValueMap(ev, tracksH, tofpOrigTrkRaw, tofpOrigTrkToken_);
  fillValueMap(ev, tracksH, assocOrigTrkRaw, assocOrigTrkToken_);
}

namespace {
  bool cmp_for_detset(const unsigned one, const unsigned two) { return one < two; };

  void find_hits_in_dets(const MTDTrackingDetSetVector& hits,
                         const Trajectory& traj,
                         const DetLayer* layer,
                         const TrajectoryStateOnSurface& tsos,
                         const float pmag2,
                         const float pathlength0,
                         const TrackSegments& trs0,
                         const float vtxTime,
                         const reco::BeamSpot& bs,
                         const float bsTimeSpread,
                         const Propagator* prop,
                         const MeasurementEstimator* estimator,
                         bool useVtxConstraint,
                         std::set<MTDHitMatchingInfo>& out) {
    pair<bool, TrajectoryStateOnSurface> comp = layer->compatible(tsos, *prop, *estimator);
    if (comp.first) {
      const vector<DetLayer::DetWithState> compDets = layer->compatibleDets(tsos, *prop, *estimator);
      LogTrace("TrackExtenderWithMTD") << "Hit search: Compatible dets " << compDets.size();
      if (!compDets.empty()) {
        for (const auto& detWithState : compDets) {
          auto range = hits.equal_range(detWithState.first->geographicalId(), cmp_for_detset);
          if (range.first == range.second) {
            LogTrace("TrackExtenderWithMTD")
                << "Hit search: no hit in DetId " << detWithState.first->geographicalId().rawId();
            continue;
          }

          auto pl = prop->propagateWithPath(tsos, detWithState.second.surface());
          if (pl.second == 0.) {
            LogTrace("TrackExtenderWithMTD")
                << "Hit search: no propagation to DetId " << detWithState.first->geographicalId().rawId();
            continue;
          }

          const float t_vtx = useVtxConstraint ? vtxTime : 0.f;

          constexpr float vtx_res = 0.008f;
          const float t_vtx_err = useVtxConstraint ? vtx_res : bsTimeSpread;

          float lastpmag2 = trs0.getSegmentPathAndMom2(0).second;

          for (auto detitr = range.first; detitr != range.second; ++detitr) {
            for (const auto& hit : *detitr) {
              auto est = estimator->estimate(detWithState.second, hit);
              if (!est.first) {
                LogTrace("TrackExtenderWithMTD")
                    << "Hit search: no compatible estimate in DetId " << detWithState.first->geographicalId().rawId()
                    << " for hit at pos (" << std::fixed << std::setw(14) << hit.globalPosition().x() << ","
                    << std::fixed << std::setw(14) << hit.globalPosition().y() << "," << std::fixed << std::setw(14)
                    << hit.globalPosition().z() << ")";
                continue;
              }

              LogTrace("TrackExtenderWithMTD")
                  << "Hit search: spatial compatibility DetId " << detWithState.first->geographicalId().rawId()
                  << " TSOS dx/dy " << std::fixed << std::setw(14)
                  << std::sqrt(detWithState.second.localError().positionError().xx()) << " " << std::fixed
                  << std::setw(14) << std::sqrt(detWithState.second.localError().positionError().yy()) << " hit dx/dy "
                  << std::fixed << std::setw(14) << std::sqrt(hit.localPositionError().xx()) << " " << std::fixed
                  << std::setw(14) << std::sqrt(hit.localPositionError().yy()) << " chi2 " << std::fixed
                  << std::setw(14) << est.second;

              TrackTofPidInfo tof = computeTrackTofPidInfo(lastpmag2,
                                                           std::abs(pl.second),
                                                           trs0,
                                                           hit.time(),
                                                           hit.timeError(),
                                                           t_vtx,
                                                           t_vtx_err,  //put vtx error by hand for the moment
                                                           false,
                                                           TofCalc::kMixd);
              MTDHitMatchingInfo mi;
              mi.hit = &hit;
              mi.estChi2 = est.second;
              mi.timeChi2 = tof.dtchi2_best;  //use the chi2 for the best matching hypothesis

              out.insert(mi);
            }
          }
        }
      }
    }
  }
}  // namespace

template <class TrackCollection>
TransientTrackingRecHit::ConstRecHitContainer TrackExtenderWithMTDT<TrackCollection>::tryBTLLayers(
    const TrajectoryStateOnSurface& tsos,
    const Trajectory& traj,
    const float pmag2,
    const float pathlength0,
    const TrackSegments& trs0,
    const MTDTrackingDetSetVector& hits,
    const MTDDetLayerGeometry* geo,
    const MagneticField* field,
    const Propagator* prop,
    const reco::BeamSpot& bs,
    const float vtxTime,
    const bool matchVertex,
    MTDHitMatchingInfo& bestHit) const {
  const vector<const DetLayer*>& layers = geo->allBTLLayers();

  TransientTrackingRecHit::ConstRecHitContainer output;
  bestHit = MTDHitMatchingInfo();
  for (const DetLayer* ilay : layers)
    fillMatchingHits(ilay, tsos, traj, pmag2, pathlength0, trs0, hits, prop, bs, vtxTime, matchVertex, output, bestHit);
  return output;
}

template <class TrackCollection>
TransientTrackingRecHit::ConstRecHitContainer TrackExtenderWithMTDT<TrackCollection>::tryETLLayers(
    const TrajectoryStateOnSurface& tsos,
    const Trajectory& traj,
    const float pmag2,
    const float pathlength0,
    const TrackSegments& trs0,
    const MTDTrackingDetSetVector& hits,
    const MTDDetLayerGeometry* geo,
    const MagneticField* field,
    const Propagator* prop,
    const reco::BeamSpot& bs,
    const float vtxTime,
    const bool matchVertex,
    MTDHitMatchingInfo& bestHit) const {
  const vector<const DetLayer*>& layers = geo->allETLLayers();

  TransientTrackingRecHit::ConstRecHitContainer output;
  bestHit = MTDHitMatchingInfo();
  for (const DetLayer* ilay : layers) {
    const BoundDisk& disk = static_cast<const ForwardDetLayer*>(ilay)->specificSurface();
    const float diskZ = disk.position().z();

    if (tsos.globalPosition().z() * diskZ < 0)
      continue;  // only propagate to the disk that's on the same side

    fillMatchingHits(ilay, tsos, traj, pmag2, pathlength0, trs0, hits, prop, bs, vtxTime, matchVertex, output, bestHit);
  }

  // the ETL hits order must be from the innermost to the outermost

  if (output.size() == 2) {
    if (std::abs(output[0]->globalPosition().z()) > std::abs(output[1]->globalPosition().z())) {
      std::reverse(output.begin(), output.end());
    }
  }
  return output;
}

template <class TrackCollection>
void TrackExtenderWithMTDT<TrackCollection>::fillMatchingHits(const DetLayer* ilay,
                                                              const TrajectoryStateOnSurface& tsos,
                                                              const Trajectory& traj,
                                                              const float pmag2,
                                                              const float pathlength0,
                                                              const TrackSegments& trs0,
                                                              const MTDTrackingDetSetVector& hits,
                                                              const Propagator* prop,
                                                              const reco::BeamSpot& bs,
                                                              const float& vtxTime,
                                                              const bool matchVertex,
                                                              TransientTrackingRecHit::ConstRecHitContainer& output,
                                                              MTDHitMatchingInfo& bestHit) const {
  std::set<MTDHitMatchingInfo> hitsInLayer;
  bool hitMatched = false;

  using namespace std::placeholders;
  auto find_hits = std::bind(find_hits_in_dets,
                             std::cref(hits),
                             std::cref(traj),
                             ilay,
                             std::cref(tsos),
                             pmag2,
                             pathlength0,
                             trs0,
                             _1,
                             std::cref(bs),
                             bsTimeSpread_,
                             prop,
                             theEstimator.get(),
                             _2,
                             std::ref(hitsInLayer));

  if (useVertex_ && matchVertex)
    find_hits(vtxTime, true);
  else
    find_hits(0, false);

  float spaceChi2Cut = ilay->isBarrel() ? btlChi2Cut_ : etlChi2Cut_;
  float timeChi2Cut = ilay->isBarrel() ? btlTimeChi2Cut_ : etlTimeChi2Cut_;

  //just take the first hit because the hits are sorted on their matching quality
  if (!hitsInLayer.empty()) {
    //check hits to pass minimum quality matching requirements
    auto const& firstHit = *hitsInLayer.begin();
    LogTrace("TrackExtenderWithMTD") << "TrackExtenderWithMTD: estChi2= " << firstHit.estChi2
                                     << " timeChi2= " << firstHit.timeChi2;
    if (firstHit.estChi2 < spaceChi2Cut && firstHit.timeChi2 < timeChi2Cut) {
      hitMatched = true;
      output.push_back(hitbuilder_->build(firstHit.hit));
      if (firstHit < bestHit)
        bestHit = firstHit;
    }
  }

  if (useVertex_ && matchVertex && !hitMatched) {
    //try a second search with beamspot hypothesis
    hitsInLayer.clear();
    find_hits(0, false);
    if (!hitsInLayer.empty()) {
      auto const& firstHit = *hitsInLayer.begin();
      LogTrace("TrackExtenderWithMTD") << "TrackExtenderWithMTD: estChi2= " << firstHit.estChi2
                                       << " timeChi2= " << firstHit.timeChi2;
      if (firstHit.timeChi2 < timeChi2Cut) {
        if (firstHit.estChi2 < spaceChi2Cut) {
          hitMatched = true;
          output.push_back(hitbuilder_->build(firstHit.hit));
          if (firstHit < bestHit)
            bestHit = firstHit;
        }
      }
    }
  }
}

//below is unfortunately ripped from other places but
//since track producer doesn't know about MTD we have to do this
template <class TrackCollection>
reco::Track TrackExtenderWithMTDT<TrackCollection>::buildTrack(const reco::TrackRef& orig,
                                                               const Trajectory& traj,
                                                               const Trajectory& trajWithMtd,
                                                               const reco::BeamSpot& bs,
                                                               const MagneticField* field,
                                                               const Propagator* thePropagator,
                                                               bool hasMTD,
                                                               float& pathLengthOut,
                                                               float& tmtdOut,
                                                               float& sigmatmtdOut,
                                                               float& tofpi,
                                                               float& tofk,
                                                               float& tofp) const {
  TrajectoryStateClosestToBeamLine tscbl;
  bool tsbcl_status = getTrajectoryStateClosestToBeamLine(traj, bs, thePropagator, tscbl);

  if (!tsbcl_status)
    return reco::Track();

  GlobalPoint v = tscbl.trackStateAtPCA().position();
  math::XYZPoint pos(v.x(), v.y(), v.z());
  GlobalVector p = tscbl.trackStateAtPCA().momentum();
  math::XYZVector mom(p.x(), p.y(), p.z());

  int ndof = traj.ndof();

  float t0 = 0.f;
  float covt0t0 = -1.f;
  pathLengthOut = -1.f;  // if there is no MTD flag the pathlength with -1
  tmtdOut = 0.f;
  sigmatmtdOut = -1.f;
  float betaOut = 0.f;
  float covbetabeta = -1.f;

  auto routput = [&]() {
    return reco::Track(traj.chiSquared(),
                       int(ndof),
                       pos,
                       mom,
                       tscbl.trackStateAtPCA().charge(),
                       tscbl.trackStateAtPCA().curvilinearError(),
                       orig->algo(),
                       reco::TrackBase::undefQuality,
                       t0,
                       betaOut,
                       covt0t0,
                       covbetabeta);
  };

  //compute path length for time backpropagation, using first MTD hit for the momentum
  if (hasMTD) {
    float pathlength;
    TrackSegments trs;
    bool validpropagation = trackPathLength(trajWithMtd, bs, thePropagator, pathlength, trs);
    float thit = 0.f;
    float thiterror = -1.f;
    bool validmtd = false;

    if (!validpropagation) {
      return routput();
    }

    uint32_t ihitcount(0), ietlcount(0);
    for (auto const& hit : trajWithMtd.measurements()) {
      if (hit.recHit()->geographicalId().det() == DetId::Forward &&
          ForwardSubdetector(hit.recHit()->geographicalId().subdetId()) == FastTime) {
        ihitcount++;
        if (MTDDetId(hit.recHit()->geographicalId()).mtdSubDetector() == MTDDetId::MTDType::ETL) {
          ietlcount++;
        }
      }
    }

    auto ihit1 = trajWithMtd.measurements().cbegin();
    if (ihitcount == 1) {
      const MTDTrackingRecHit* mtdhit = static_cast<const MTDTrackingRecHit*>((*ihit1).recHit()->hit());
      thit = mtdhit->time();
      thiterror = mtdhit->timeError();
      validmtd = true;
    } else if (ihitcount == 2 && ietlcount == 2) {
      std::pair<float, float> lastStep = trs.getSegmentPathAndMom2(0);
      float etlpathlength = std::abs(lastStep.first * c_cm_ns);
      //
      // The information of the two ETL hits is combined and attributed to the innermost hit
      //
      if (etlpathlength == 0.f) {
        validpropagation = false;
      } else {
        pathlength -= etlpathlength;
        trs.removeFirstSegment();
        const MTDTrackingRecHit* mtdhit1 = static_cast<const MTDTrackingRecHit*>((*ihit1).recHit()->hit());
        const MTDTrackingRecHit* mtdhit2 = static_cast<const MTDTrackingRecHit*>((*(ihit1 + 1)).recHit()->hit());
        TrackTofPidInfo tofInfo = computeTrackTofPidInfo(
            lastStep.second, etlpathlength, trs, mtdhit1->time(), mtdhit1->timeError(), 0.f, 0.f, true, TofCalc::kCost);
        //
        // Protect against incompatible times
        //
        float err1 = tofInfo.dterror * tofInfo.dterror;
        float err2 = mtdhit2->timeError() * mtdhit2->timeError();
        if (cms_rounding::roundIfNear0(err1) == 0.f || cms_rounding::roundIfNear0(err2) == 0.f) {
          edm::LogError("TrackExtenderWithMTD")
              << "MTD tracking hits with zero time uncertainty: " << err1 << " " << err2;
        } else {
          if ((tofInfo.dt - mtdhit2->time()) * (tofInfo.dt - mtdhit2->time()) < (err1 + err2) * etlTimeChi2Cut_) {
            //
            // Subtract the ETL time of flight from the outermost measurement, and combine it in a weighted average with the innermost
            // the mass ambiguity related uncertainty on the time of flight is added as an additional uncertainty
            //
            err1 = 1.f / err1;
            err2 = 1.f / err2;
            thiterror = 1.f / (err1 + err2);
            thit = (tofInfo.dt * err1 + mtdhit2->time() * err2) * thiterror;
            thiterror = std::sqrt(thiterror);
            LogDebug("TrackExtenderWithMTD")
                << "TrackExtenderWithMTD: p trk = " << p.mag() << " ETL hits times/errors: " << mtdhit1->time()
                << " +/- " << mtdhit1->timeError() << " , " << mtdhit2->time() << " +/- " << mtdhit2->timeError()
                << " extrapolated time1: " << tofInfo.dt << " +/- " << tofInfo.dterror << " average = " << thit
                << " +/- " << thiterror;
            validmtd = true;
          }
	  // if back extrapolated time of the outermost measurement not compatible with the innermost, keep the one with smallest error 
	  else {
	    if ( err1 <= err2 ) {
	      thit = tofInfo.dt;
	      thiterror = tofInfo.dterror;
	      validmtd = true;
	    }
	    else {
	      thit = mtdhit2->time();
	      thiterror = mtdhit2->timeError();
	      validmtd = true;
	    }
	  }
        }
      }
    } else {
      edm::LogInfo("TrackExtenderWithMTD")
          << "MTD hits #" << ihitcount << "ETL hits #" << ietlcount << " anomalous pattern, skipping...";
    }

    if (validmtd && validpropagation) {
      //here add the PID uncertainty for later use in the 1st step of 4D vtx reconstruction
      TrackTofPidInfo tofInfo =
          computeTrackTofPidInfo(p.mag2(), pathlength, trs, thit, thiterror, 0.f, 0.f, true, TofCalc::kSegm);
      pathLengthOut = pathlength;  // set path length if we've got a timing hit
      tmtdOut = thit;
      sigmatmtdOut = thiterror;
      t0 = tofInfo.dt;
      covt0t0 = tofInfo.dterror * tofInfo.dterror;
      betaOut = tofInfo.beta_pi;
      covbetabeta = tofInfo.betaerror * tofInfo.betaerror;
      tofpi = tofInfo.dt_pi;
      tofk = tofInfo.dt_k;
      tofp = tofInfo.dt_p;
    }
  }

  return routput();
}

template <class TrackCollection>
reco::TrackExtra TrackExtenderWithMTDT<TrackCollection>::buildTrackExtra(const Trajectory& trajectory) const {
  static const string metname = "TrackExtenderWithMTD";

  const Trajectory::RecHitContainer transRecHits = trajectory.recHits();

  // put the collection of TrackingRecHit in the event

  // sets the outermost and innermost TSOSs
  // ToDo: validation for track states with MTD
  TrajectoryStateOnSurface outerTSOS;
  TrajectoryStateOnSurface innerTSOS;
  unsigned int innerId = 0, outerId = 0;
  TrajectoryMeasurement::ConstRecHitPointer outerRecHit;
  DetId outerDetId;

  if (trajectory.direction() == alongMomentum) {
    LogTrace(metname) << "alongMomentum";
    outerTSOS = trajectory.lastMeasurement().updatedState();
    innerTSOS = trajectory.firstMeasurement().updatedState();
    outerId = trajectory.lastMeasurement().recHit()->geographicalId().rawId();
    innerId = trajectory.firstMeasurement().recHit()->geographicalId().rawId();
    outerRecHit = trajectory.lastMeasurement().recHit();
    outerDetId = trajectory.lastMeasurement().recHit()->geographicalId();
  } else if (trajectory.direction() == oppositeToMomentum) {
    LogTrace(metname) << "oppositeToMomentum";
    outerTSOS = trajectory.firstMeasurement().updatedState();
    innerTSOS = trajectory.lastMeasurement().updatedState();
    outerId = trajectory.firstMeasurement().recHit()->geographicalId().rawId();
    innerId = trajectory.lastMeasurement().recHit()->geographicalId().rawId();
    outerRecHit = trajectory.firstMeasurement().recHit();
    outerDetId = trajectory.firstMeasurement().recHit()->geographicalId();
  } else
    LogError(metname) << "Wrong propagation direction!";

  const GeomDet* outerDet = gtg_->idToDet(outerDetId);
  GlobalPoint outerTSOSPos = outerTSOS.globalParameters().position();
  bool inside = outerDet->surface().bounds().inside(outerDet->toLocal(outerTSOSPos));

  GlobalPoint hitPos =
      (outerRecHit->isValid()) ? outerRecHit->globalPosition() : outerTSOS.globalParameters().position();

  if (!inside) {
    LogTrace(metname) << "The Global Muon outerMostMeasurementState is not compatible with the recHit detector!"
                      << " Setting outerMost postition to recHit position if recHit isValid: "
                      << outerRecHit->isValid();
    LogTrace(metname) << "From " << outerTSOSPos << " to " << hitPos;
  }

  //build the TrackExtra
  GlobalPoint v = (inside) ? outerTSOSPos : hitPos;
  GlobalVector p = outerTSOS.globalParameters().momentum();
  math::XYZPoint outpos(v.x(), v.y(), v.z());
  math::XYZVector outmom(p.x(), p.y(), p.z());

  v = innerTSOS.globalParameters().position();
  p = innerTSOS.globalParameters().momentum();
  math::XYZPoint inpos(v.x(), v.y(), v.z());
  math::XYZVector inmom(p.x(), p.y(), p.z());

  reco::TrackExtra trackExtra(outpos,
                              outmom,
                              true,
                              inpos,
                              inmom,
                              true,
                              outerTSOS.curvilinearError(),
                              outerId,
                              innerTSOS.curvilinearError(),
                              innerId,
                              trajectory.direction(),
                              trajectory.seedRef());

  return trackExtra;
}

template <class TrackCollection>
string TrackExtenderWithMTDT<TrackCollection>::dumpLayer(const DetLayer* layer) const {
  stringstream output;

  const BoundSurface* sur = nullptr;
  const BoundCylinder* bc = nullptr;
  const BoundDisk* bd = nullptr;

  sur = &(layer->surface());
  if ((bc = dynamic_cast<const BoundCylinder*>(sur))) {
    output << "  Cylinder of radius: " << bc->radius() << endl;
  } else if ((bd = dynamic_cast<const BoundDisk*>(sur))) {
    output << "  Disk at: " << bd->position().z() << endl;
  }
  return output.str();
}

//define this as a plug-in
#include <FWCore/Framework/interface/MakerMacros.h>
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
typedef TrackExtenderWithMTDT<reco::TrackCollection> TrackExtenderWithMTD;

DEFINE_FWK_MODULE(TrackExtenderWithMTD);
