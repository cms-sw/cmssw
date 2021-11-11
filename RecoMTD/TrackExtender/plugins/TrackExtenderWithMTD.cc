#define EDM_ML_DEBUG

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
  constexpr double c_cm_ns = geant_units::operators::convertMmToCm(CLHEP::c_light);  // [mm/ns] -> [cm/ns]
  constexpr double c_inv = 1.0 / c_cm_ns;

  class MTDHitMatchingInfo {
  public:
    MTDHitMatchingInfo() {
      hit = nullptr;
      estChi2 = std::numeric_limits<double>::max();
      timeChi2 = std::numeric_limits<double>::max();
    }

    //Operator used to sort the hits while performing the matching step at the MTD
    inline bool operator<(const MTDHitMatchingInfo& m2) const {
      //only for good matching in time use estChi2, otherwise use mostly time compatibility
      constexpr double chi2_cut = 10.;
      constexpr double low_weight = 3.;
      constexpr double high_weight = 8.;
      if (timeChi2 < chi2_cut && m2.timeChi2 < chi2_cut)
        return chi2(low_weight) < m2.chi2(low_weight);
      else
        return chi2(high_weight) < m2.chi2(high_weight);
    }

    inline double chi2(double timeWeight = 1.) const { return estChi2 + timeWeight * timeChi2; }

    const MTDTrackingRecHit* hit;
    double estChi2;
    double timeChi2;
  };

  class TrackSegments {
  public:
    TrackSegments() {
      nSegment_ = 0;
      segmentPathOvc_.clear();
      segmentMom2_.clear();
    }

    inline size_t addSegment(double tPath, double tMom2) {
      segmentPathOvc_.emplace_back(tPath * c_inv);
      segmentMom2_.emplace_back(tMom2);
      nSegment_++;

      LogTrace("TrackExtenderWithMTD") << "addSegment # " << nSegment_ << " s = " << tPath
                                       << " p = " << std::sqrt(tMom2);

      return nSegment_;
    }

    inline const double computeTof(double mass_inv2) {
      double tof(0.);
      for (size_t iSeg = 0; iSeg < nSegment_; iSeg++) {
        double gammasq = 1. + segmentMom2_[iSeg] * mass_inv2;
        double beta = std::sqrt(1. - 1. / gammasq);
        tof += segmentPathOvc_[iSeg] / beta;

        LogTrace("TrackExtenderWithMTD") << " TOF Segment # " << iSeg + 1 << " p = " << std::sqrt(segmentMom2_[iSeg])
                                         << " tof = " << tof;
      }

      return tof;
    }

    inline size_t getSize() const { return nSegment_; }

    inline size_t removeFirstSegment() {
      if (nSegment_ > 0) {
        segmentPathOvc_.erase(segmentPathOvc_.begin());
        segmentMom2_.erase(segmentMom2_.begin());
        nSegment_--;
      }
      return nSegment_;
    }

    inline std::pair<double, double> getSegment(size_t iSegment) const {
      if (iSegment >= nSegment_) {
        throw cms::Exception("TrackExtenderWithMTD") << "Requesting non existing track segment #" << iSegment;
      }
      return std::make_pair(segmentPathOvc_[iSegment], segmentMom2_[iSegment]);
    }

    size_t nSegment_;
    std::vector<double> segmentPathOvc_;
    std::vector<double> segmentMom2_;
  };

  struct TrackTofPidInfo {
    double tmtd;
    double tmtderror;
    double pathlength;

    double betaerror;

    double dt;
    double dterror;
    double dtchi2;

    double dt_best;
    double dterror_best;
    double dtchi2_best;

    double gammasq_pi;
    double beta_pi;
    double dt_pi;

    double gammasq_k;
    double beta_k;
    double dt_k;

    double gammasq_p;
    double beta_p;
    double dt_p;

    double prob_pi;
    double prob_k;
    double prob_p;
  };

  enum class TofCalc { cost = 1, segm = 2, mixd = 3 };

  const TrackTofPidInfo computeTrackTofPidInfo(double magp2,
                                               double length,
                                               TrackSegments trs,
                                               double t_mtd,
                                               double t_mtderr,
                                               double t_vtx,
                                               double t_vtx_err,
                                               bool addPIDError = true,
                                               TofCalc choice = TofCalc::cost) {
    constexpr double m_pi = 0.13957018;
    constexpr double m_pi_inv2 = 1.0 / m_pi / m_pi;
    constexpr double m_k = 0.493677;
    constexpr double m_k_inv2 = 1.0 / m_k / m_k;
    constexpr double m_p = 0.9382720813;
    constexpr double m_p_inv2 = 1.0 / m_p / m_p;

    TrackTofPidInfo tofpid;

    tofpid.tmtd = t_mtd;
    tofpid.tmtderror = t_mtderr;
    tofpid.pathlength = length;

    tofpid.gammasq_pi = 1. + magp2 * m_pi_inv2;
    tofpid.beta_pi = std::sqrt(1. - 1. / tofpid.gammasq_pi);
    switch (choice) {
      case TofCalc::cost:
        tofpid.dt_pi = tofpid.pathlength / tofpid.beta_pi * c_inv;
        break;
      case TofCalc::segm:
        tofpid.dt_pi = trs.computeTof(m_pi_inv2);
        break;
      case TofCalc::mixd:
        tofpid.dt_pi = trs.computeTof(m_pi_inv2) + tofpid.pathlength / tofpid.beta_pi * c_inv;
        break;
      default:
        tofpid.dt_pi = 1;
        break;
    }

    tofpid.gammasq_k = 1. + magp2 * m_k_inv2;
    tofpid.beta_k = std::sqrt(1. - 1. / tofpid.gammasq_k);
    switch (choice) {
      case TofCalc::cost:
        tofpid.dt_k = tofpid.pathlength / tofpid.beta_k * c_inv;
        break;
      case TofCalc::segm:
        tofpid.dt_k = trs.computeTof(m_k_inv2);
        break;
      case TofCalc::mixd:
        tofpid.dt_k = trs.computeTof(m_k_inv2) + tofpid.pathlength / tofpid.beta_k * c_inv;
        break;
      default:
        tofpid.dt_k = 1;
        break;
    }

    tofpid.gammasq_p = 1. + magp2 * m_p_inv2;
    tofpid.beta_p = std::sqrt(1. - 1. / tofpid.gammasq_p);
    switch (choice) {
      case TofCalc::cost:
        tofpid.dt_p = tofpid.pathlength / tofpid.beta_p * c_inv;
        break;
      case TofCalc::segm:
        tofpid.dt_p = trs.computeTof(m_p_inv2);
        break;
      case TofCalc::mixd:
        tofpid.dt_p = trs.computeTof(m_p_inv2) + tofpid.pathlength / tofpid.beta_p * c_inv;
        break;
      default:
        tofpid.dt_p = 1;
        break;
    }

    tofpid.dt = tofpid.tmtd - tofpid.dt_pi - t_vtx;  //assume by default the pi hypothesis
    tofpid.dterror = sqrt(tofpid.tmtderror * tofpid.tmtderror + t_vtx_err * t_vtx_err);
    tofpid.betaerror = 0;
    if (addPIDError) {
      tofpid.dterror =
          sqrt(tofpid.dterror * tofpid.dterror + (tofpid.dt_p - tofpid.dt_pi) * (tofpid.dt_p - tofpid.dt_pi));
      tofpid.betaerror = tofpid.beta_p - tofpid.beta_pi;
    }

    tofpid.dtchi2 = (tofpid.dt * tofpid.dt) / (tofpid.dterror * tofpid.dterror);

    tofpid.dt_best = tofpid.dt;
    tofpid.dterror_best = tofpid.dterror;
    tofpid.dtchi2_best = tofpid.dtchi2;

    tofpid.prob_pi = -1.;
    tofpid.prob_k = -1.;
    tofpid.prob_p = -1.;

    if (!addPIDError) {
      //*TODO* deal with heavier nucleons and/or BSM case here?
      double chi2_pi = tofpid.dtchi2;
      double chi2_k =
          (tofpid.tmtd - tofpid.dt_k - t_vtx) * (tofpid.tmtd - tofpid.dt_k - t_vtx) / (tofpid.dterror * tofpid.dterror);
      double chi2_p =
          (tofpid.tmtd - tofpid.dt_p - t_vtx) * (tofpid.tmtd - tofpid.dt_p - t_vtx) / (tofpid.dterror * tofpid.dterror);

      double rawprob_pi = exp(-0.5 * chi2_pi);
      double rawprob_k = exp(-0.5 * chi2_k);
      double rawprob_p = exp(-0.5 * chi2_p);
      double normprob = 1. / (rawprob_pi + rawprob_k + rawprob_p);

      tofpid.prob_pi = rawprob_pi * normprob;
      tofpid.prob_k = rawprob_k * normprob;
      tofpid.prob_p = rawprob_p * normprob;

      double prob_heavy = 1. - tofpid.prob_pi;
      constexpr double heavy_threshold = 0.75;

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
                       double& pathlength,
                       TrackSegments& trs) {
    pathlength = 0.;

    bool validpropagation = true;
    double oldp = traj.measurements().begin()->updatedState().globalMomentum().mag();
    double pathlength1 = 0.;
    double pathlength2 = 0.;

    //add pathlength layer by layer
    for (auto it = traj.measurements().begin(); it != traj.measurements().end() - 1; ++it) {
      const auto& propresult = thePropagator->propagateWithPath(it->updatedState(), (it + 1)->updatedState().surface());
      double layerpathlength = std::abs(propresult.second);
      if (layerpathlength == 0.) {
        validpropagation = false;
      }
      pathlength1 += layerpathlength;
      trs.addSegment(layerpathlength, (it + 1)->updatedState().globalMomentum().mag2());
      LogTrace("TrackExtenderWithMTD") << "TSOS " << std::fixed << std::setw(4) << trs.getSize() << " R_i "
                                       << std::fixed << std::setw(14) << it->updatedState().globalPosition().perp()
                                       << " z_i " << std::fixed << std::setw(14)
                                       << it->updatedState().globalPosition().z() << " R_e " << std::fixed
                                       << std::setw(14) << (it + 1)->updatedState().globalPosition().perp() << " z_e "
                                       << std::fixed << std::setw(14) << (it + 1)->updatedState().globalPosition().z()
                                       << " p " << std::fixed << std::setw(14)
                                       << (it + 1)->updatedState().globalMomentum().mag() << " dp " << std::fixed
                                       << std::setw(14) << (it + 1)->updatedState().globalMomentum().mag() - oldp;
      oldp = (it + 1)->updatedState().globalMomentum().mag();
    }

    //add distance from bs to first measurement
    auto const& tscblPCA = tscbl.trackStateAtPCA();
    auto const& aSurface = traj.direction() == alongMomentum ? traj.firstMeasurement().updatedState().surface()
                                                             : traj.lastMeasurement().updatedState().surface();
    pathlength2 = thePropagator->propagateWithPath(tscblPCA, aSurface).second;
    if (pathlength2 == 0.) {
      validpropagation = false;
    }
    pathlength = pathlength1 + pathlength2;
    trs.addSegment(pathlength2, tscblPCA.momentum().mag2());
    LogTrace("TrackExtenderWithMTD") << "TSOS " << std::fixed << std::setw(4) << trs.getSize() << " R_e " << std::fixed
                                     << std::setw(14) << tscblPCA.position().perp() << " z_e " << std::fixed
                                     << std::setw(14) << tscblPCA.position().z() << " p " << std::fixed << std::setw(14)
                                     << tscblPCA.momentum().mag() << " dp " << std::fixed << std::setw(14)
                                     << tscblPCA.momentum().mag() - oldp;
    return validpropagation;
  }

  bool trackPathLength(const Trajectory& traj,
                       const reco::BeamSpot& bs,
                       const Propagator* thePropagator,
                       double& pathlength,
                       TrackSegments& trs) {
    pathlength = 0.;

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
                                                             const double,
                                                             const double,
                                                             const TrackSegments&,
                                                             const MTDTrackingDetSetVector&,
                                                             const MTDDetLayerGeometry*,
                                                             const MagneticField* field,
                                                             const Propagator* prop,
                                                             const reco::BeamSpot& bs,
                                                             const double vtxTime,
                                                             const bool matchVertex,
                                                             MTDHitMatchingInfo& bestHit) const;

  TransientTrackingRecHit::ConstRecHitContainer tryETLLayers(const TrajectoryStateOnSurface&,
                                                             const Trajectory& traj,
                                                             const double,
                                                             const double,
                                                             const TrackSegments&,
                                                             const MTDTrackingDetSetVector&,
                                                             const MTDDetLayerGeometry*,
                                                             const MagneticField* field,
                                                             const Propagator* prop,
                                                             const reco::BeamSpot& bs,
                                                             const double vtxTime,
                                                             const bool matchVertex,
                                                             MTDHitMatchingInfo& bestHit) const;

  void fillMatchingHits(const DetLayer*,
                        const TrajectoryStateOnSurface&,
                        const Trajectory&,
                        const double,
                        const double,
                        const TrackSegments&,
                        const MTDTrackingDetSetVector&,
                        const Propagator*,
                        const reco::BeamSpot&,
                        const double&,
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

  reco::Track buildTrack(const reco::Track&,
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
  edm::EDPutToken btlMatchChi2Token;
  edm::EDPutToken etlMatchChi2Token;
  edm::EDPutToken btlMatchTimeChi2Token;
  edm::EDPutToken etlMatchTimeChi2Token;
  edm::EDPutToken npixBarrelToken;
  edm::EDPutToken npixEndcapToken;
  edm::EDPutToken pOrigTrkToken;
  edm::EDPutToken betaOrigTrkToken;
  edm::EDPutToken t0OrigTrkToken;
  edm::EDPutToken sigmat0OrigTrkToken;
  edm::EDPutToken pathLengthOrigTrkToken;
  edm::EDPutToken tmtdOrigTrkToken;
  edm::EDPutToken sigmatmtdOrigTrkToken;
  edm::EDPutToken tofpiOrigTrkToken;
  edm::EDPutToken tofkOrigTrkToken;
  edm::EDPutToken tofpOrigTrkToken;
  edm::EDPutToken assocOrigTrkToken;

  edm::EDGetTokenT<InputCollection> tracksToken_;
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

  btlMatchChi2Token = produces<edm::ValueMap<float>>("btlMatchChi2");
  etlMatchChi2Token = produces<edm::ValueMap<float>>("etlMatchChi2");
  btlMatchTimeChi2Token = produces<edm::ValueMap<float>>("btlMatchTimeChi2");
  etlMatchTimeChi2Token = produces<edm::ValueMap<float>>("etlMatchTimeChi2");
  npixBarrelToken = produces<edm::ValueMap<int>>("npixBarrel");
  npixEndcapToken = produces<edm::ValueMap<int>>("npixEndcap");
  pOrigTrkToken = produces<edm::ValueMap<float>>("generalTrackp");
  betaOrigTrkToken = produces<edm::ValueMap<float>>("generalTrackBeta");
  t0OrigTrkToken = produces<edm::ValueMap<float>>("generalTrackt0");
  sigmat0OrigTrkToken = produces<edm::ValueMap<float>>("generalTracksigmat0");
  pathLengthOrigTrkToken = produces<edm::ValueMap<float>>("generalTrackPathLength");
  tmtdOrigTrkToken = produces<edm::ValueMap<float>>("generalTracktmtd");
  sigmatmtdOrigTrkToken = produces<edm::ValueMap<float>>("generalTracksigmatmtd");
  tofpiOrigTrkToken = produces<edm::ValueMap<float>>("generalTrackTofPi");
  tofkOrigTrkToken = produces<edm::ValueMap<float>>("generalTrackTofK");
  tofpOrigTrkToken = produces<edm::ValueMap<float>>("generalTrackTofP");
  assocOrigTrkToken = produces<edm::ValueMap<int>>("generalTrackassoc");

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
  const auto& tracks = *tracksH;

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

  double vtxTime = 0.;
  if (useVertex_) {
    if (useSimVertex_ && genPV) {
      vtxTime = genPV->t();
    } else if (pv)
      vtxTime = pv->t();  //already in ns
  }

  std::vector<unsigned> track_indices;
  unsigned itrack = 0;

  for (const auto& track : tracks) {
    double trackVtxTime = 0.;
    if (useVertex_) {
      double dz;
      if (useSimVertex_)
        dz = std::abs(track.dz(math::XYZPoint(*genPV)));
      else
        dz = std::abs(track.dz(pv->position()));

      if (dz < dzCut_)
        trackVtxTime = vtxTime;
    }

    reco::TransientTrack ttrack(track, magfield.product(), gtg_);
    const auto& trajs = theTransformer->transform(track);
    auto thits = theTransformer->getTransientRecHits(ttrack);
    TransientTrackingRecHit::ConstRecHitContainer mtdthits;
    MTDHitMatchingInfo mBTL, mETL;
    if (!trajs.empty()) {
      // get the outermost trajectory point on the track
      TrajectoryStateOnSurface tsos = builder_->build(track).outermostMeasurementState();
      TrajectoryStateClosestToBeamLine tscbl;
      bool tscbl_status = getTrajectoryStateClosestToBeamLine(trajs.front(), bs, prop, tscbl);

      if (tscbl_status) {
        double pmag2 = tscbl.trackStateAtPCA().momentum().mag2();
        double pathlength0;
        TrackSegments trs0;
        trackPathLength(trajs.front(), tscbl, prop, pathlength0, trs0);

        const auto& btlhits = tryBTLLayers(tsos,
                                           trajs.front(),
                                           pmag2,
                                           pathlength0,
                                           trs0,
                                           hits,
                                           geo.product(),
                                           magfield.product(),
                                           prop,
                                           bs,
                                           trackVtxTime,
                                           trackVtxTime != 0.,
                                           mBTL);
        mtdthits.insert(mtdthits.end(), btlhits.begin(), btlhits.end());

        // in the future this should include an intermediate refit before propagating to the ETL
        // for now it is ok
        const auto& etlhits = tryETLLayers(tsos,
                                           trajs.front(),
                                           pmag2,
                                           pathlength0,
                                           trs0,
                                           hits,
                                           geo.product(),
                                           magfield.product(),
                                           prop,
                                           bs,
                                           trackVtxTime,
                                           trackVtxTime != 0.,
                                           mETL);
        mtdthits.insert(mtdthits.end(), etlhits.begin(), etlhits.end());
      }
    }  //!trajs.empty()

    auto ordering = checkRecHitsOrdering(thits);
    if (ordering == RefitDirection::insideOut) {
      thits.insert(thits.end(), mtdthits.begin(), mtdthits.end());
    } else {
      std::reverse(mtdthits.begin(), mtdthits.end());
      mtdthits.insert(mtdthits.end(), thits.begin(), thits.end());
      thits.swap(mtdthits);
    }

    const auto& trajwithmtd = mtdthits.empty() ? trajs : theTransformer->transform(ttrack, thits);
    float pMap = 0.f, betaMap = 0.f, t0Map = 0.f, sigmat0Map = -1.f, pathLengthMap = -1.f, tmtdMap = 0.f,
          sigmatmtdMap = -1.f, tofpiMap = 0.f, tofkMap = 0.f, tofpMap = 0.f;
    int iMap = -1;

    for (const auto& trj : trajwithmtd) {
      const auto& thetrj = (updateTraj_ ? trj : trajs.front());
      float pathLength = 0.f, tmtd = 0.f, sigmatmtd = -1.f, tofpi = 0.f, tofk = 0.f, tofp = 0.f;
      LogTrace("TrackExtenderWithMTD") << "Refit track " << itrack << " p/pT = " << track.p() << " " << track.pt();
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
        btlMatchChi2.push_back(mBTL.hit ? mBTL.estChi2 : -1);
        etlMatchChi2.push_back(mETL.hit ? mETL.estChi2 : -1);
        btlMatchTimeChi2.push_back(mBTL.hit ? mBTL.timeChi2 : -1);
        etlMatchTimeChi2.push_back(mETL.hit ? mETL.timeChi2 : -1);
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
        backtrack.setExtra((updateExtra_ ? extraRef : track.extra()));
        for (unsigned ihit = hitsstart; ihit < hitsend; ++ihit) {
          backtrack.appendHitPattern((*outhits)[ihit], ttopo);
        }
        npixBarrel.push_back(backtrack.hitPattern().numberOfValidPixelBarrelHits());
        npixEndcap.push_back(backtrack.hitPattern().numberOfValidPixelEndcapHits());
        LogTrace("TrackExtenderWithMTD") << "tmtd " << tmtdMap << " +/- " << sigmatmtdMap << " t0 " << t0Map << " +/- " << sigmat0Map << " tof pi/K/p " << tofpiMap << " " << tofkMap << " " << tofpMap;
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
      btlMatchChi2.push_back(-1.);
      etlMatchChi2.push_back(-1.);
      btlMatchTimeChi2.push_back(-1.);
      etlMatchTimeChi2.push_back(-1.);
      npixBarrel.push_back(-1.);
      npixEndcap.push_back(-1.);
    }

    ++itrack;
  }

  auto outTrksHandle = ev.put(std::move(output));
  ev.put(std::move(extras));
  ev.put(std::move(outhits));

  fillValueMap(ev, tracksH, btlMatchChi2, btlMatchChi2Token);
  fillValueMap(ev, tracksH, etlMatchChi2, etlMatchChi2Token);
  fillValueMap(ev, tracksH, btlMatchTimeChi2, btlMatchTimeChi2Token);
  fillValueMap(ev, tracksH, etlMatchTimeChi2, etlMatchTimeChi2Token);
  fillValueMap(ev, tracksH, npixBarrel, npixBarrelToken);
  fillValueMap(ev, tracksH, npixEndcap, npixEndcapToken);
  fillValueMap(ev, tracksH, pOrigTrkRaw, pOrigTrkToken);
  fillValueMap(ev, tracksH, betaOrigTrkRaw, betaOrigTrkToken);
  fillValueMap(ev, tracksH, t0OrigTrkRaw, t0OrigTrkToken);
  fillValueMap(ev, tracksH, sigmat0OrigTrkRaw, sigmat0OrigTrkToken);
  fillValueMap(ev, tracksH, pathLengthsOrigTrkRaw, pathLengthOrigTrkToken);
  fillValueMap(ev, tracksH, tmtdOrigTrkRaw, tmtdOrigTrkToken);
  fillValueMap(ev, tracksH, sigmatmtdOrigTrkRaw, sigmatmtdOrigTrkToken);
  fillValueMap(ev, tracksH, tofpiOrigTrkRaw, tofpiOrigTrkToken);
  fillValueMap(ev, tracksH, tofkOrigTrkRaw, tofkOrigTrkToken);
  fillValueMap(ev, tracksH, tofpOrigTrkRaw, tofpOrigTrkToken);
  fillValueMap(ev, tracksH, assocOrigTrkRaw, assocOrigTrkToken);
}

namespace {
  bool cmp_for_detset(const unsigned one, const unsigned two) { return one < two; };

  void find_hits_in_dets(const MTDTrackingDetSetVector& hits,
                         const Trajectory& traj,
                         const DetLayer* layer,
                         const TrajectoryStateOnSurface& tsos,
                         const double pmag2,
                         const double pathlength0,
                         const TrackSegments& trs0,
                         const double vtxTime,
                         const reco::BeamSpot& bs,
                         const float bsTimeSpread,
                         const Propagator* prop,
                         const MeasurementEstimator* estimator,
                         bool useVtxConstraint,
                         std::set<MTDHitMatchingInfo>& out) {
    pair<bool, TrajectoryStateOnSurface> comp = layer->compatible(tsos, *prop, *estimator);
    if (comp.first) {
      const vector<DetLayer::DetWithState> compDets = layer->compatibleDets(tsos, *prop, *estimator);
      if (!compDets.empty()) {
        for (const auto& detWithState : compDets) {
          auto range = hits.equal_range(detWithState.first->geographicalId(), cmp_for_detset);
          if (range.first == range.second)
            continue;

          auto pl = prop->propagateWithPath(tsos, detWithState.second.surface());
          if (pl.second == 0.)
            continue;

          //const double tot_pl = pathlength0 + std::abs(pl.second);
          const double t_vtx = useVtxConstraint ? vtxTime : 0.;

          constexpr double vtx_res = 0.008;
          const double t_vtx_err = useVtxConstraint ? vtx_res : bsTimeSpread;

          //constexpr double t_res_manual = 0.035;

          double lastpmag2 = trs0.getSegment(0).second;

          for (auto detitr = range.first; detitr != range.second; ++detitr) {
            for (const auto& hit : *detitr) {
              auto est = estimator->estimate(detWithState.second, hit);
              if (!est.first)
                continue;

              //TrackTofPidInfo tof = computeTrackTofPidInfo(pmag2,
                                                           //tot_pl,
                                                           //trs0,
                                                           //hit.time(),
                                                           //t_res_manual,  //put hit error by hand for the moment
                                                           //t_vtx,
                                                           //t_vtx_err,  //put vtx error by hand for the moment
                                                           //false,
                                                           //TofCalc::cost);

              LogTrace("TrackExtenderWithMTD") << "Cand hit t = " << hit.time() << " +/- " << hit.timeError() << " p/lastp " << std::sqrt(pmag2) << " / " << std::sqrt(lastpmag2);
              TrackTofPidInfo tof = computeTrackTofPidInfo(lastpmag2,
                                                           std::abs(pl.second),
                                                           trs0,
                                                           hit.time(),
                                                           hit.timeError(),
                                                           t_vtx,
                                                           t_vtx_err,  //put vtx error by hand for the moment
                                                           false,
                                                           TofCalc::mixd);
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
    const double pmag2,
    const double pathlength0,
    const TrackSegments& trs0,
    const MTDTrackingDetSetVector& hits,
    const MTDDetLayerGeometry* geo,
    const MagneticField* field,
    const Propagator* prop,
    const reco::BeamSpot& bs,
    const double vtxTime,
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
    const double pmag2,
    const double pathlength0,
    const TrackSegments& trs0,
    const MTDTrackingDetSetVector& hits,
    const MTDDetLayerGeometry* geo,
    const MagneticField* field,
    const Propagator* prop,
    const reco::BeamSpot& bs,
    const double vtxTime,
    const bool matchVertex,
    MTDHitMatchingInfo& bestHit) const {
  const vector<const DetLayer*>& layers = geo->allETLLayers();

  TransientTrackingRecHit::ConstRecHitContainer output;
  bestHit = MTDHitMatchingInfo();
  for (const DetLayer* ilay : layers) {
    const BoundDisk& disk = static_cast<const ForwardDetLayer*>(ilay)->specificSurface();
    const double diskZ = disk.position().z();

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
                                                              const double pmag2,
                                                              const double pathlength0,
                                                              const TrackSegments& trs0,
                                                              const MTDTrackingDetSetVector& hits,
                                                              const Propagator* prop,
                                                              const reco::BeamSpot& bs,
                                                              const double& vtxTime,
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

  //just take the first hit because the hits are sorted on their matching quality
  if (!hitsInLayer.empty()) {
    //check hits to pass minimum quality matching requirements
    auto const& firstHit = *hitsInLayer.begin();
    if (firstHit.estChi2 < etlChi2Cut_ && firstHit.timeChi2 < etlTimeChi2Cut_) {
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
      if (firstHit.timeChi2 < etlTimeChi2Cut_) {
        if (firstHit.estChi2 < etlChi2Cut_) {
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
reco::Track TrackExtenderWithMTDT<TrackCollection>::buildTrack(const reco::Track& orig,
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

  double t0 = 0.;
  double covt0t0 = -1.;
  pathLengthOut = -1.f;  // if there is no MTD flag the pathlength with -1
  tmtdOut = 0.f;
  sigmatmtdOut = -1.f;
  double betaOut = 0.;
  double covbetabeta = -1.;

  auto routput = [&]() {
    return reco::Track(traj.chiSquared(),
                       int(ndof),
                       pos,
                       mom,
                       tscbl.trackStateAtPCA().charge(),
                       tscbl.trackStateAtPCA().curvilinearError(),
                       orig.algo(),
                       reco::TrackBase::undefQuality,
                       t0,
                       betaOut,
                       covt0t0,
                       covbetabeta);
  };

  //compute path length for time backpropagation, using first MTD hit for the momentum
  if (hasMTD) {
    double pathlength;
    TrackSegments trs;
    bool validpropagation = trackPathLength(trajWithMtd, bs, thePropagator, pathlength, trs);
    double thit = 0.;
    double thiterror = -1.;
    bool validmtd = false;

    if (!validpropagation) {
      return routput();
    }

    size_t ihitcount(0), ietlcount(0);
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
      std::pair <double,double> lastStep = trs.getSegment(0);
      double etlpathlength = std::abs(lastStep.first/c_inv);
      //
      // The information of the two ETL hits is combined and attributed to the innermost hit
      //
      if (etlpathlength == 0.) {
        validpropagation = false;
      } else {
        pathlength -= etlpathlength;
        trs.removeFirstSegment();
        const MTDTrackingRecHit* mtdhit1 = static_cast<const MTDTrackingRecHit*>((*ihit1).recHit()->hit());
        const MTDTrackingRecHit* mtdhit2 = static_cast<const MTDTrackingRecHit*>((*(ihit1 + 1)).recHit()->hit());
        TrackTofPidInfo tofInfo = computeTrackTofPidInfo(
          lastStep.second, etlpathlength, trs, mtdhit1->time(), mtdhit1->timeError(), 0., 0., true, TofCalc::cost);
        //
        // Protect against incompatible times
        //
        double err1 = tofInfo.dterror * tofInfo.dterror;
        double err2 = mtdhit2->timeError() * mtdhit2->timeError();
        if (cms_rounding::roundIfNear0(err1) == 0. || cms_rounding::roundIfNear0(err2) == 0.) {
          edm::LogError("TrackExtenderWithMTD")
              << "MTD tracking hits with zero time uncertainty: " << err1 << " " << err2;
        } else {
          if ((tofInfo.dt - mtdhit2->time()) * (tofInfo.dt - mtdhit2->time()) < (err1 + err2) * etlTimeChi2Cut_) {
            //
            // Subtract the ETL time of flight from the outermost measurement, and combine it in a weighted average with the innermost
            // the mass ambiguity related uncertainty on the time of flight is added as an additional uncertainty
            //
            err1 = 1. / err1;
            err2 = 1. / err2;
            thiterror = 1. / (err1 + err2);
            thit = (tofInfo.dt * err1 + mtdhit2->time() * err2) * thiterror;
            thiterror = std::sqrt(thiterror);
            LogDebug("TrackExtenderWithMTD") << "p trk = " << p.mag() << " ETL hits times/errors: " << mtdhit1->time()
                                             << " +/- " << mtdhit1->timeError() << " , " << mtdhit2->time() << " +/- "
                                             << mtdhit2->timeError() << " extrapolated time1: " << tofInfo.dt << " +/- "
                                             << tofInfo.dterror << " average = " << thit << " +/- " << thiterror;
            validmtd = true;
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
          computeTrackTofPidInfo(p.mag2(), pathlength, trs, thit, thiterror, 0., 0., true, TofCalc::segm);
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
