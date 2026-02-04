#include <sstream>
#include <format>

#include "RecoMTD/TrackExtender/plugins/MuonExtenderWithMTD.h"

#include <CLHEP/Units/GlobalPhysicalConstants.h>

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include "DataFormats/ForwardDetId/interface/MTDChannelIdentifier.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Rounding.h"
#include "DataFormats/TrackerRecHit2D/interface/MTDTrackingRecHit.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonTopologies/interface/Topology.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoMTD/DetLayers/interface/MTDDetLayerGeometry.h"
#include "RecoMTD/DetLayers/interface/MTDTrayBarrelLayer.h"
#include "RecoMTD/Records/interface/MTDRecoGeometryRecord.h"
#include "RecoMTD/TransientTrackingRecHit/interface/MTDTransientTrackingRecHitBuilder.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"
#include "RecoMuon/Navigation/interface/MuonNavigationSchool.h"
#include "RecoMuon/Navigation/interface/MuonNavigationPrinter.h"
#include "RecoTracker/TransientTrackingRecHit/interface/Traj2TrackHits.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderWithPropagator.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TrackRefitter/interface/RefitDirection.h"

using namespace std;
using namespace edm;
using namespace reco;
using namespace mtdtof;

namespace mtdtof {

  // Asummes 100% muon identification
  const MuonTofPidInfo computeMuonTofPidInfo(float magp2,
                                             float length,
                                             TrackSegments trs,
                                             float t_mtd,
                                             float t_mtderr,
                                             float t_vtx,
                                             float t_vtx_err,
                                             bool addPIDError,
                                             TofCalc choice,
                                             SigmaTofCalc sigma_choice) {
    constexpr float m_mu = 0.10565837f;
    constexpr float m_mu_inv2 = 1.0f / m_mu / m_mu;

    MuonTofPidInfo tofpid;

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

    auto sigmadeltat = [&](const float mass_inv2) {
      float res(1.f);
      switch (sigma_choice) {
        case SigmaTofCalc::kCost:
          // sigma(t) = sigma(p) * |dt/dp| = sigma(p) * DeltaL/c * m^2 / (p^2 * E)
          res = tofpid.pathlength * c_inv * trs.segmentSigmaMom_[trs.nSegment_ - 1] /
                (magp2 * sqrt(magp2 + 1 / mass_inv2) * mass_inv2);
          break;
        case SigmaTofCalc::kSegm:
          res = trs.computeSigmaTof(mass_inv2);
          break;
        case SigmaTofCalc::kMixd:
          float res1 = tofpid.pathlength * c_inv * trs.segmentSigmaMom_[trs.nSegment_ - 1] /
                       (magp2 * sqrt(magp2 + 1 / mass_inv2) * mass_inv2);
          float res2 = trs.computeSigmaTof(mass_inv2);
          res = sqrt(res1 * res1 + res2 * res2 + 2 * res1 * res2);
      }

      return res;
    };

    tofpid.gammasq_mu = 1.f + magp2 * m_mu_inv2;
    tofpid.beta_mu = std::sqrt(1.f - 1.f / tofpid.gammasq_mu);
    tofpid.dt_mu = deltat(m_mu_inv2, tofpid.beta_mu);
    tofpid.sigma_dt_mu = sigmadeltat(m_mu_inv2);

    tofpid.dt = tofpid.tmtd - tofpid.dt_mu - t_vtx;
    tofpid.betaerror = 0.f;
    tofpid.dterror2 =
        tofpid.tmtderror * tofpid.tmtderror + t_vtx_err * t_vtx_err + tofpid.sigma_dt_mu * tofpid.sigma_dt_mu;
    tofpid.dterror = sqrt(tofpid.dterror2);

    tofpid.dtchi2 = (tofpid.dt * tofpid.dt) / tofpid.dterror2;

    tofpid.dt_best = tofpid.dt;
    tofpid.dterror_best = tofpid.dterror;
    tofpid.dtchi2_best = tofpid.dtchi2;

    return tofpid;
  };

  bool muonPathLength(const Trajectory& traj,
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
      if (it->recHit()->isValid())
        if (it->recHit()->geographicalId().det() == DetId::Muon)
          continue;

      const auto& propresult = thePropagator->propagateWithPath(it->updatedState(), (it + 1)->updatedState().surface());
      float layerpathlength = std::abs(propresult.second);
      if (layerpathlength == 0.f) {
        validpropagation = false;
      }
      pathlength1 += layerpathlength;

      // sigma(p) from curvilinear error (on q/p)
      float sigma_p = sqrt((it + 1)->updatedState().curvilinearError().matrix()(0, 0)) *
                      (it + 1)->updatedState().globalMomentum().mag2();

      trs.addSegment(layerpathlength, (it + 1)->updatedState().globalMomentum().mag2(), sigma_p);

      LogTrace("MuonExtenderWithMTD") << "TSOS " << std::fixed << std::setw(4) << trs.size() << " R_i " << std::fixed
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

    float sigma_p = sqrt(tscblPCA.curvilinearError().matrix()(0, 0)) * tscblPCA.momentum().mag2();

    trs.addSegment(pathlength2, tscblPCA.momentum().mag2(), sigma_p);

    LogTrace("MuonExtenderWithMTD") << "TSOS " << std::fixed << std::setw(4) << trs.size() << " R_e " << std::fixed
                                    << std::setw(14) << tscblPCA.position().perp() << " z_e " << std::fixed
                                    << std::setw(14) << tscblPCA.position().z() << " p " << std::fixed << std::setw(14)
                                    << tscblPCA.momentum().mag() << " dp " << std::fixed << std::setw(14)
                                    << tscblPCA.momentum().mag() - oldp << " sigma_p = " << std::fixed << std::setw(14)
                                    << sigma_p << " sigma_p/p = " << std::fixed << std::setw(14)
                                    << sigma_p / tscblPCA.momentum().mag() * 100 << " %";

    return validpropagation;
  };

  bool muonPathLength(const Trajectory& traj,
                      const reco::BeamSpot& bs,
                      const Propagator* thePropagator,
                      float& pathlength,
                      TrackSegments& trs) {
    pathlength = 0.f;

    TrajectoryStateClosestToBeamLine tscbl;
    bool tscbl_status = getTrajectoryStateClosestToBeamLine(traj, bs, thePropagator, tscbl);

    if (!tscbl_status)
      return false;

    return muonPathLength(traj, tscbl, thePropagator, pathlength, trs);
  };
  void find_hits_in_dets_muon(const MTDTrackingDetSetVector& hits,
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
                              std::set<MTDHitMatchingInfo>& out) {
    pair<bool, TrajectoryStateOnSurface> comp = layer->compatible(tsos, *prop, *estimator);
    if (comp.first) {
      const vector<DetLayer::DetWithState> compDets = layer->compatibleDets(tsos, *prop, *estimator);
      LogTrace("MuonExtenderWithMTD") << "Hit search: Compatible dets " << compDets.size();
      if (!compDets.empty()) {
        for (const auto& detWithState : compDets) {
          auto range = hits.equal_range(detWithState.first->geographicalId(), cmp_for_detset);
          if (range.first == range.second) {
            LogTrace("MuonExtenderWithMTD")
                << "Hit search: no hit in DetId " << detWithState.first->geographicalId().rawId();
            continue;
          }

          auto pl = prop->propagateWithPath(tsos, detWithState.second.surface());
          if (pl.second == 0.) {
            LogTrace("MuonExtenderWithMTD")
                << "Hit search: no propagation to DetId " << detWithState.first->geographicalId().rawId();
            continue;
          }

          const float t_vtx = useVtxConstraint ? vtxTime : 0.f;

          const float t_vtx_err = useVtxConstraint ? vtxTimeError : bsTimeSpread;

          float lastpmag2 = trs0.getSegmentPathAndMom2(0).second;

          for (auto detitr = range.first; detitr != range.second; ++detitr) {
            for (const auto& hit : *detitr) {
              auto est = estimator->estimate(detWithState.second, hit);
              if (!est.first) {
                LogTrace("MuonExtenderWithMTD")
                    << "Hit search: no compatible estimate in DetId " << detWithState.first->geographicalId().rawId()
                    << " for hit at pos (" << std::fixed << std::setw(14) << hit.globalPosition().x() << ","
                    << std::fixed << std::setw(14) << hit.globalPosition().y() << "," << std::fixed << std::setw(14)
                    << hit.globalPosition().z() << ")";
                continue;
              }

              LogTrace("MuonExtenderWithMTD")
                  << "Hit search: spatial compatibility DetId " << detWithState.first->geographicalId().rawId()
                  << " TSOS dx/dy " << std::fixed << std::setw(14)
                  << std::sqrt(detWithState.second.localError().positionError().xx()) << " " << std::fixed
                  << std::setw(14) << std::sqrt(detWithState.second.localError().positionError().yy()) << " hit dx/dy "
                  << std::fixed << std::setw(14) << std::sqrt(hit.localPositionError().xx()) << " " << std::fixed
                  << std::setw(14) << std::sqrt(hit.localPositionError().yy()) << " chi2 " << std::fixed
                  << std::setw(14) << est.second;

              MuonTofPidInfo tof = computeMuonTofPidInfo(lastpmag2,
                                                         std::abs(pl.second),
                                                         trs0,
                                                         hit.time(),
                                                         hit.timeError(),
                                                         t_vtx,
                                                         t_vtx_err,  //put vtx error by hand for the moment
                                                         false,
                                                         TofCalc::kMixd,
                                                         SigmaTofCalc::kMixd);
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
  };
}  // namespace mtdtof

// MuonBaseExtenderWithMTD ---------

MuonBaseExtenderWithMTD::MuonBaseExtenderWithMTD(const edm::ParameterSet& iConfig) : BaseExtenderWithMTD(iConfig) {}
MuonBaseExtenderWithMTD::~MuonBaseExtenderWithMTD() = default;

void MuonBaseExtenderWithMTD::fillMatchingHits(const DetLayer* ilay,
                                               const TrajectoryStateOnSurface& tsos,
                                               const Trajectory& traj,
                                               const float pmag2,
                                               const float pathlength0,
                                               const TrackSegments& trs0,
                                               const MTDTrackingDetSetVector& hits,
                                               const Propagator* prop,
                                               const reco::BeamSpot& bs,
                                               const float& vtxTime,
                                               const float& vtxTimeError,
                                               TransientTrackingRecHit::ConstRecHitContainer& output,
                                               MTDHitMatchingInfo& bestHit) const {
  std::set<MTDHitMatchingInfo> hitsInLayer;
  bool hitMatched = false;

  using namespace std::placeholders;
  auto find_hits = std::bind(find_hits_in_dets_muon,
                             std::cref(hits),
                             std::cref(traj),
                             ilay,
                             std::cref(tsos),
                             pmag2,
                             pathlength0,
                             trs0,
                             _1,
                             _2,
                             _3,
                             std::cref(bs),
                             bsTimeSpread_,
                             prop,
                             theEstimator.get(),
                             std::ref(hitsInLayer));

  bool matchVertex = vtxTimeError > 0.f;
  if (useVertex_ && matchVertex) {
    find_hits(vtxTime, vtxTimeError, true);
  } else {
    find_hits(0, 0, false);
  }

  float spaceChi2Cut = ilay->isBarrel() ? btlChi2Cut_ : etlChi2Cut_;
  float timeChi2Cut = ilay->isBarrel() ? btlTimeChi2Cut_ : etlTimeChi2Cut_;

  //just take the first hit because the hits are sorted on their matching quality
  if (!hitsInLayer.empty()) {
    //check hits to pass minimum quality matching requirements
    auto const& firstHit = *hitsInLayer.begin();
    LogTrace("MuonExtenderWithMTD") << "MuonExtenderWithMTD: matching trial 1: estChi2= " << firstHit.estChi2
                                    << " timeChi2= " << firstHit.timeChi2;
    if (firstHit.estChi2 < spaceChi2Cut && firstHit.timeChi2 < timeChi2Cut) {
      hitMatched = true;
      output.push_back(getHitBuilder()->build(firstHit.hit));
      if (firstHit < bestHit)
        bestHit = firstHit;
    }
  }

  if (useVertex_ && matchVertex && !hitMatched) {
    //try a second search with beamspot hypothesis
    hitsInLayer.clear();
    find_hits(0, 0, false);
    if (!hitsInLayer.empty()) {
      auto const& firstHit = *hitsInLayer.begin();
      LogTrace("MuonExtenderWithMTD") << "MuonExtenderWithMTD: matching trial 2: estChi2= " << firstHit.estChi2
                                      << " timeChi2= " << firstHit.timeChi2;
      if (firstHit.timeChi2 < timeChi2Cut) {
        if (firstHit.estChi2 < spaceChi2Cut) {
          hitMatched = true;
          output.push_back(getHitBuilder()->build(firstHit.hit));
          if (firstHit < bestHit)
            bestHit = firstHit;
        }
      }
    }
  }
#ifdef EDM_ML_DEBUG
  if (hitMatched) {
    LogTrace("MuonExtenderWithMTD") << "MuonExtenderWithMTD: matched hit with time: " << bestHit.hit->time() << " +/- "
                                    << bestHit.hit->timeError();
  } else {
    LogTrace("MuonExtenderWithMTD") << "MuonExtenderWithMTD: no matched hit";
  }
#endif
}

reco::Track MuonBaseExtenderWithMTD::buildTrack(const reco::TrackBase::TrackAlgorithm origAlgo,
                                                const Trajectory& traj,
                                                const Trajectory& trajWithMtd,
                                                const reco::BeamSpot& bs,
                                                const MagneticField* field,
                                                const Propagator* thePropagator,
                                                bool hasMTD,
                                                float& pathLengthOut,
                                                float& tmtdOut,
                                                float& sigmatmtdOut,
                                                GlobalPoint& tmtdPosOut,
                                                float& tofmu,
                                                float& sigmatofmu) const {
  TrajectoryStateClosestToBeamLine tscbl;
  bool tsbcl_status = getTrajectoryStateClosestToBeamLine(traj, bs, thePropagator, tscbl);

  if (!tsbcl_status)
    return reco::Track();

  GlobalPoint v = tscbl.trackStateAtPCA().position();
  math::XYZPoint pos(v.x(), v.y(), v.z());
  GlobalVector p = tscbl.trackStateAtPCA().momentum();
  math::XYZVector mom(p.x(), p.y(), p.z());

  int ndof = trajWithMtd.ndof();

  float t0 = 0.f;
  float covt0t0 = -1.f;
  pathLengthOut = -1.f;  // if there is no MTD flag the pathlength with -1
  tmtdOut = 0.f;
  sigmatmtdOut = -1.f;
  float betaOut = 0.f;
  float covbetabeta = -1.f;

  auto routput = [&]() {
    return reco::Track(trajWithMtd.chiSquared(),
                       int(ndof),
                       pos,
                       mom,
                       tscbl.trackStateAtPCA().charge(),
                       tscbl.trackStateAtPCA().curvilinearError(),
                       origAlgo,
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
    bool validpropagation = muonPathLength(trajWithMtd, bs, thePropagator, pathlength, trs);
    float thit = 0.f;
    float thiterror = -1.f;
    GlobalPoint thitpos{0., 0., 0.};
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

    LogTrace("MuonExtenderWithMTD") << "MuonExtenderWithMTD: selected #hits " << ihitcount << " from ETL " << ietlcount;

    // The last measurement is not always the MTD Hit for muons
    auto ihit1 = trajWithMtd.measurements().cbegin();
    for (auto it = trajWithMtd.measurements().cbegin(); it != trajWithMtd.measurements().cend(); ++it) {
      const auto& hit = it->recHit();
      if (hit->geographicalId().det() == DetId::Forward &&
          ForwardSubdetector(hit->geographicalId().subdetId()) == FastTime) {
        ihit1 = it;
        break;
      }
    }
    if (ihitcount == 1) {
      const MTDTrackingRecHit* mtdhit = static_cast<const MTDTrackingRecHit*>((*ihit1).recHit()->hit());
      thit = mtdhit->time();
      thiterror = mtdhit->timeError();
      thitpos = mtdhit->globalPosition();
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

        MuonTofPidInfo tofInfo = computeMuonTofPidInfo(lastStep.second,
                                                       etlpathlength,
                                                       trs,
                                                       mtdhit1->time(),
                                                       mtdhit1->timeError(),
                                                       0.f,
                                                       0.f,
                                                       true,
                                                       TofCalc::kCost,
                                                       SigmaTofCalc::kCost);
        //
        // Protect against incompatible times
        //
        float err1 = tofInfo.dterror2;
        float err2 = mtdhit2->timeError() * mtdhit2->timeError();
        if (cms_rounding::roundIfNear0(err1) == 0.f || cms_rounding::roundIfNear0(err2) == 0.f) {
          edm::LogError("MuonExtenderWithMTD")
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
            thitpos = mtdhit2->globalPosition();
            LogTrace("MuonExtenderWithMTD")
                << "MuonExtenderWithMTD: p trk = " << p.mag() << " ETL hits times/errors: 1) " << mtdhit1->time()
                << " +/- " << mtdhit1->timeError() << " , 2) " << mtdhit2->time() << " +/- " << mtdhit2->timeError()
                << " extrapolated time1: " << tofInfo.dt << " +/- " << tofInfo.dterror << " average = " << thit
                << " +/- " << thiterror << "\n    hit1 pos: " << mtdhit1->globalPosition()
                << " hit2 pos: " << mtdhit2->globalPosition() << " etl path length " << etlpathlength << std::endl;

            validmtd = true;
          } else {
            // if back extrapolated time of the outermost measurement not compatible with the innermost, keep the one with smallest error
            if (err1 <= err2) {
              thit = tofInfo.dt;
              thiterror = tofInfo.dterror;
              validmtd = true;
            } else {
              thit = mtdhit2->time();
              thiterror = mtdhit2->timeError();
              validmtd = true;
            }
          }
        }
      }
    } else {
      edm::LogInfo("MuonExtenderWithMTD")
          << "MTD hits #" << ihitcount << "ETL hits #" << ietlcount << " anomalous pattern, skipping...";
    }

    if (validmtd && validpropagation) {
      //here add the PID uncertainty for later use in the 1st step of 4D vtx reconstruction

      MuonTofPidInfo tofInfo = computeMuonTofPidInfo(
          p.mag2(), pathlength, trs, thit, thiterror, 0.f, 0.f, true, TofCalc::kSegm, SigmaTofCalc::kCost);

      pathLengthOut = pathlength;  // set path length if we've got a timing hit
      tmtdOut = thit;
      sigmatmtdOut = thiterror;
      tmtdPosOut = thitpos;
      t0 = tofInfo.dt;
      covt0t0 = tofInfo.dterror2;
      betaOut = tofInfo.beta_mu;
      covbetabeta = tofInfo.betaerror * tofInfo.betaerror;
      tofmu = tofInfo.dt_mu;
      sigmatofmu = tofInfo.sigma_dt_mu;
    }
  }
  return routput();
}

template <typename T1>
MuonExtenderWithMTDT<T1>::MuonExtenderWithMTDT(const ParameterSet& iConfig)
    : muonToken_(consumes<T1>(iConfig.getParameter<edm::InputTag>("muonSrc"))),
      hitsToken_(consumes<MTDTrackingDetSetVector>(iConfig.getParameter<edm::InputTag>("hitsSrc"))),
      bsToken_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpotSrc"))),
      updateTraj_(iConfig.getParameter<bool>("updateTrackTrajectory")),
      updateExtra_(iConfig.getParameter<bool>("updateTrackExtra")),
      updatePattern_(iConfig.getParameter<bool>("updateTrackHitPattern")),
      mtdRecHitBuilder_(iConfig.getParameter<std::string>("MTDRecHitBuilder")),
      propagator_(iConfig.getParameter<std::string>("Propagator")),
      transientTrackBuilder_(iConfig.getParameter<std::string>("TransientTrackBuilder")),
      useVertex_(iConfig.getParameter<bool>("useVertex")) {
  baseMTDExtender_ = std::make_unique<MuonBaseExtenderWithMTD>(iConfig);

  if (useVertex_) {
    vtxToken_ = consumes<VertexCollection>(iConfig.getParameter<edm::InputTag>("vtxSrc"));
  }

  theTransformer = std::make_unique<TrackTransformer>(iConfig.getParameterSet("TrackTransformer"), consumesCollector());

  btlMatchChi2Token_ = produces<edm::ValueMap<float>>("btlMatchChi2");
  etlMatchChi2Token_ = produces<edm::ValueMap<float>>("etlMatchChi2");
  btlMatchTimeChi2Token_ = produces<edm::ValueMap<float>>("btlMatchTimeChi2");
  etlMatchTimeChi2Token_ = produces<edm::ValueMap<float>>("etlMatchTimeChi2");
  npixBarrelToken_ = produces<edm::ValueMap<int>>("npixBarrel");
  npixEndcapToken_ = produces<edm::ValueMap<int>>("npixEndcap");
  outermostHitPositionToken_ = produces<edm::ValueMap<float>>("generalTrackOutermostHitPosition");
  pOrigTrkToken_ = produces<edm::ValueMap<float>>("generalTrackp");
  betaOrigTrkToken_ = produces<edm::ValueMap<float>>("generalTrackBeta");
  t0OrigTrkToken_ = produces<edm::ValueMap<float>>("generalTrackt0");
  sigmat0OrigTrkToken_ = produces<edm::ValueMap<float>>("generalTracksigmat0");
  pathLengthOrigTrkToken_ = produces<edm::ValueMap<float>>("generalTrackPathLength");
  tmtdOrigTrkToken_ = produces<edm::ValueMap<float>>("generalTracktmtd");
  sigmatmtdOrigTrkToken_ = produces<edm::ValueMap<float>>("generalTracksigmatmtd");
  tmtdPosOrigTrkToken_ = produces<edm::ValueMap<GlobalPoint>>("generalTrackmtdpos");
  tofmuOrigTrkToken_ = produces<edm::ValueMap<float>>("generalTrackTofMu");
  sigmatofmuOrigTrkToken_ = produces<edm::ValueMap<float>>("generalTrackSigmaTofMu");
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

template <typename T1>
template <class H, class T>
void MuonExtenderWithMTDT<T1>::fillValueMap(edm::Event& iEvent,
                                            const H& handle,
                                            const std::vector<T>& vec,
                                            const edm::EDPutToken& token) const {
  auto out = std::make_unique<edm::ValueMap<T>>();
  typename edm::ValueMap<T>::Filler filler(*out);
  filler.insert(handle, vec.begin(), vec.end());
  filler.fill();
  iEvent.put(token, std::move(out));
}

template <typename T1>
void MuonExtenderWithMTDT<T1>::produce(edm::Event& ev, const edm::EventSetup& es) {
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

  baseMTDExtender_->setParameters(&*hitbuilder_, &*gtg_);

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
  std::vector<float> outermostHitPosition;
  std::vector<float> pOrigTrkRaw;
  std::vector<float> betaOrigTrkRaw;
  std::vector<float> t0OrigTrkRaw;
  std::vector<float> sigmat0OrigTrkRaw;
  std::vector<float> pathLengthsOrigTrkRaw;
  std::vector<float> tmtdOrigTrkRaw;
  std::vector<float> sigmatmtdOrigTrkRaw;
  std::vector<GlobalPoint> tmtdPosOrigTrkRaw;
  std::vector<float> tofmuOrigTrkRaw;
  std::vector<float> sigmatofmuOrigTrkRaw;
  std::vector<int> assocOrigTrkRaw;

  edm::Handle<T1> muons;
  ev.getByToken(muonToken_, muons);

  //MTD hits DetSet
  const auto& hits = ev.get(hitsToken_);

  //beam spot
  const auto& bs = ev.get(bsToken_);

  bool vtxConstraint(false);

  VertexCollection vtxs;
  if (useVertex_) {
    vtxs = ev.get(vtxToken_);
    if (!vtxs.empty()) {
      vtxConstraint = true;
    }
  }

  std::vector<unsigned> track_indices;
  unsigned imu = 0;

  for (const auto& itmuon : *muons) {
    if (!(itmuon.track().isNonnull()))
      continue;  // Require a track to be extrapolated to the MTD

    LogTrace("MuonExtenderWithMTD") << "MuonExtenderWithMTD: extrapolating track " << imu << " p/pT = " << itmuon.p()
                                    << " " << itmuon.pt() << " eta = " << itmuon.eta();
    LogTrace("MuonExtenderWithMTD") << "MuonExtenderWithMTD: sigma_p = "
                                    << sqrt(itmuon.track()->covariance()(0, 0)) * itmuon.track()->p2()
                                    << " sigma_p/p = "
                                    << sqrt(itmuon.track()->covariance()(0, 0)) * itmuon.track()->p() * 100 << " %";

    float trackVtxTime = 0.f;
    float trackVtxTimeError = 0.f;
    if (vtxConstraint) {
      for (const auto& vtx : vtxs) {
        for (size_t itrk = 0; itrk < vtx.tracksSize(); itrk++) {
          if (itmuon.track() == vtx.trackRefAt(itrk).castTo<TrackRef>()) {
            trackVtxTime = vtx.t();
            trackVtxTimeError = vtx.tError();
            break;
          }
        }
      }
    }

    reco::TransientTrack ttrack(itmuon.track(), magfield.product(), gtg_);
    auto thits = theTransformer->getTransientRecHits(ttrack);
    const auto trajVec = theTransformer->transform(ttrack, thits);
    const Trajectory& trajs = trajVec.front();

    TransientTrackingRecHit::ConstRecHitContainer mtdthits;
    MTDHitMatchingInfo mBTL, mETL;

    TrajectoryStateOnSurface tsos = ttrack.outermostMeasurementState();

    if (!trajs.measurements().empty()) {
      // Loop over measurements and get the last one in the tracker
      auto ordering = baseMTDExtender_->checkRecHitsOrdering(thits);
      if (ordering == RefitDirection::insideOut) {
        for (auto it = trajs.measurements().rbegin(); it != trajs.measurements().rend(); ++it) {
          const auto& hit = it->recHit();
          if (!hit->isValid())
            continue;
          if (hit->geographicalId().det() == DetId::Tracker) {
            // Got last tracker hit
            tsos = it->updatedState();  // this is the outer TSOS in tracker
          } else {
            continue;
          }
        }
      } else {
        for (auto it = trajs.measurements().begin(); it != trajs.measurements().end(); ++it) {
          const auto& hit = it->recHit();
          if (!hit->isValid())
            continue;
          if (hit->geographicalId().det() == DetId::Tracker) {
            // Got last tracker hit
            tsos = it->updatedState();  // this is the outer TSOS in tracker
          } else {
            continue;
          }
        }
      }
    }

    TrajectoryStateClosestToBeamLine tscbl;
    bool tscbl_status = getTrajectoryStateClosestToBeamLine(trajs, bs, prop, tscbl);

    if (tscbl_status) {
      float pmag2 = tscbl.trackStateAtPCA().momentum().mag2();
      float pathlength0;
      TrackSegments trs0;
      muonPathLength(trajs, tscbl, prop, pathlength0, trs0);

      const auto& btlhits = baseMTDExtender_->tryBTLLayers(tsos,
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
                                                           trackVtxTimeError,
                                                           mBTL);
      mtdthits.insert(mtdthits.end(), btlhits.begin(), btlhits.end());

      // in the future this should include an intermediate refit before propagating to the ETL
      // for now it is ok
      const auto& etlhits = baseMTDExtender_->tryETLLayers(tsos,
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
                                                           trackVtxTimeError,
                                                           mETL);
      mtdthits.insert(mtdthits.end(), etlhits.begin(), etlhits.end());
    }
#ifdef EDM_ML_DEBUG
    else {
      LogTrace("MuonExtenderWithMTD") << "Failing getTrajectoryStateClosestToBeamLine, no search for hits in MTD!";
    }
#endif

    // Logic to embed MTD Hits between TRK and Muon hits
    auto ordering = baseMTDExtender_->checkRecHitsOrdering(thits);
    TransientTrackingRecHit::ConstRecHitContainer innerthits;
    TransientTrackingRecHit::ConstRecHitContainer outerthits;
    if (ordering == RefitDirection::insideOut) {
      for (auto const& hit : thits) {
        if (hit->geographicalId().det() == DetId::Tracker) {
          innerthits.push_back(hit);
        } else if (hit->geographicalId().det() == DetId::Muon) {
          outerthits.push_back(hit);
        } else {
          LogTrace("MuonExtenderWithMTD") << "Where is this hit coming from?";
        }
      }
      thits = innerthits;
      thits.insert(thits.end(), mtdthits.begin(), mtdthits.end());
      thits.insert(thits.end(), outerthits.begin(), outerthits.end());
    } else {
      for (auto const& hit : thits) {
        if (hit->geographicalId().det() == DetId::Tracker) {
          innerthits.push_back(hit);
        } else if (hit->geographicalId().det() == DetId::Muon) {
          outerthits.push_back(hit);
        } else {
          LogTrace("MuonExtenderWithMTD") << "Where is this hit coming from?";
        }
      }
      std::reverse(mtdthits.begin(), mtdthits.end());
      outerthits.insert(outerthits.end(), mtdthits.begin(), mtdthits.end());
      outerthits.insert(outerthits.end(), innerthits.begin(), innerthits.end());
      thits.swap(outerthits);
    }

    const auto& trajwithmtd = theTransformer->transform(ttrack, thits);
    float pMap = 0.f, betaMap = 0.f, t0Map = 0.f, sigmat0Map = -1.f, pathLengthMap = -1.f, tmtdMap = 0.f,
          sigmatmtdMap = -1.f, tofmuMap = 0.f, sigmatofmuMap = -1.f;
    GlobalPoint tmtdPosMap{0., 0., 0.};
    int iMap = -1;

    for (const auto& trj : trajwithmtd) {
      const auto& thetrj = (updateTraj_ ? trj : trajs);
      float pathLength = 0.f, tmtd = 0.f, sigmatmtd = -1.f, tofmu = 0.f, sigmatofmu = -1.f;
      GlobalPoint tmtdPos{0., 0., 0.};
      LogTrace("MuonExtenderWithMTD") << "MuonExtenderWithMTD: refit track " << imu << " p/pT = " << itmuon.track()->p()
                                      << " " << itmuon.track()->pt() << " eta = " << itmuon.track()->eta();

      reco::Track result = baseMTDExtender_->buildTrack(itmuon.track()->algo(),
                                                        thetrj,
                                                        trj,
                                                        bs,
                                                        magfield.product(),
                                                        prop,
                                                        !mtdthits.empty(),
                                                        pathLength,
                                                        tmtd,
                                                        sigmatmtd,
                                                        tmtdPos,
                                                        tofmu,
                                                        sigmatofmu);

      if (result.ndof() >= 0) {
        /// setup the track extras
        reco::TrackExtra::TrajParams trajParams;
        reco::TrackExtra::Chi2sFive chi2s;
        size_t hitsstart = outhits->size();
        //t2t(trj, *outhits, trajParams, chi2s);
        if (updatePattern_) {
          t2t(trj, *outhits, trajParams, chi2s);  // this fills the output hit collection
        } else {
          t2t(thetrj, *outhits, trajParams, chi2s);
        }
        size_t hitsend = outhits->size();
        extras->push_back(baseMTDExtender_->buildTrackExtra(
            trj));  // always push back the fully built extra, update by setting in track
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
        tmtdPosMap = tmtdPos;
        auto& backtrack = output->back();
        iMap = output->size() - 1;
        pMap = backtrack.p();
        betaMap = backtrack.beta();
        t0Map = backtrack.t0();
        sigmat0Map = std::copysign(std::sqrt(std::abs(backtrack.covt0t0())), backtrack.covt0t0());
        tofmuMap = tofmu;
        sigmatofmuMap = sigmatofmu;
        reco::TrackExtraRef extraRef(extrasRefProd, extras->size() - 1);
        backtrack.setExtra((updateExtra_ ? extraRef : itmuon.track()->extra()));
        for (unsigned ihit = hitsstart; ihit < hitsend; ++ihit) {
          backtrack.appendHitPattern((*outhits)[ihit], ttopo);
        }
#ifdef EDM_ML_DEBUG
        LogTrace("MuonExtenderWithMTD") << "MuonExtenderWithMTD: hit pattern of refitted track";
        for (int i = 0; i < backtrack.hitPattern().numberOfAllHits(reco::HitPattern::TRACK_HITS); i++) {
          backtrack.hitPattern().printHitPattern(reco::HitPattern::TRACK_HITS, i, std::cout);
        }
        LogTrace("MuonExtenderWithMTD") << "MuonExtenderWithMTD: missing hit pattern of refitted track";
        for (int i = 0; i < backtrack.hitPattern().numberOfAllHits(reco::HitPattern::MISSING_INNER_HITS); i++) {
          backtrack.hitPattern().printHitPattern(reco::HitPattern::MISSING_INNER_HITS, i, std::cout);
        }
#endif
        npixBarrel.push_back(backtrack.hitPattern().numberOfValidPixelBarrelHits());
        npixEndcap.push_back(backtrack.hitPattern().numberOfValidPixelEndcapHits());

        if (mBTL.hit || mETL.hit) {
          outermostHitPosition.push_back(
              mBTL.hit ? (float)(*itmuon.track()).outerRadius()
                       : (float)(*itmuon.track()).outerZ());  // save R of the outermost hit for BTL, z for ETL.
        } else {
          outermostHitPosition.push_back(std::abs(itmuon.track()->eta()) < trackMaxBtlEta_
                                             ? (float)(*itmuon.track()).outerRadius()
                                             : (float)(*itmuon.track()).outerZ());
        }

        LogTrace("MuonExtenderWithMTD") << "MuonExtenderWithMTD: tmtd " << tmtdMap << " +/- " << sigmatmtdMap << " t0 "
                                        << t0Map << " +/- " << sigmat0Map << " tof mu " << tofmuMap << "+/-"
                                        << std::format("{:0.2g}", sigmatofmuMap) << " ("
                                        << std::format("{:0.2g}", sigmatofmuMap / tofmuMap * 100) << "%) ";

      } else {
        LogTrace("MuonExtenderWithMTD") << "Error in the MTD track refitting. This should not happen";
      }
    }

    pOrigTrkRaw.push_back(pMap);
    betaOrigTrkRaw.push_back(betaMap);
    t0OrigTrkRaw.push_back(t0Map);
    sigmat0OrigTrkRaw.push_back(sigmat0Map);
    pathLengthsOrigTrkRaw.push_back(pathLengthMap);
    tmtdOrigTrkRaw.push_back(tmtdMap);
    sigmatmtdOrigTrkRaw.push_back(sigmatmtdMap);
    tmtdPosOrigTrkRaw.push_back(tmtdPosMap);
    tofmuOrigTrkRaw.push_back(tofmuMap);
    sigmatofmuOrigTrkRaw.push_back(sigmatofmuMap);
    assocOrigTrkRaw.push_back(iMap);

    if (iMap == -1) {
      btlMatchChi2.push_back(-1.f);
      etlMatchChi2.push_back(-1.f);
      btlMatchTimeChi2.push_back(-1.f);
      etlMatchTimeChi2.push_back(-1.f);
      npixBarrel.push_back(-1.f);
      npixEndcap.push_back(-1.f);
      outermostHitPosition.push_back(0.);
    }

    ++imu;
  }

  ev.put(std::move(output));
  ev.put(std::move(extras));
  ev.put(std::move(outhits));

  fillValueMap(ev, muons, btlMatchChi2, btlMatchChi2Token_);
  fillValueMap(ev, muons, etlMatchChi2, etlMatchChi2Token_);
  fillValueMap(ev, muons, btlMatchTimeChi2, btlMatchTimeChi2Token_);
  fillValueMap(ev, muons, etlMatchTimeChi2, etlMatchTimeChi2Token_);
  fillValueMap(ev, muons, npixBarrel, npixBarrelToken_);
  fillValueMap(ev, muons, npixEndcap, npixEndcapToken_);
  fillValueMap(ev, muons, outermostHitPosition, outermostHitPositionToken_);
  fillValueMap(ev, muons, pOrigTrkRaw, pOrigTrkToken_);
  fillValueMap(ev, muons, betaOrigTrkRaw, betaOrigTrkToken_);
  fillValueMap(ev, muons, t0OrigTrkRaw, t0OrigTrkToken_);
  fillValueMap(ev, muons, sigmat0OrigTrkRaw, sigmat0OrigTrkToken_);
  fillValueMap(ev, muons, pathLengthsOrigTrkRaw, pathLengthOrigTrkToken_);
  fillValueMap(ev, muons, tmtdOrigTrkRaw, tmtdOrigTrkToken_);
  fillValueMap(ev, muons, sigmatmtdOrigTrkRaw, sigmatmtdOrigTrkToken_);
  fillValueMap(ev, muons, tmtdPosOrigTrkRaw, tmtdPosOrigTrkToken_);
  fillValueMap(ev, muons, tofmuOrigTrkRaw, tofmuOrigTrkToken_);
  fillValueMap(ev, muons, sigmatofmuOrigTrkRaw, sigmatofmuOrigTrkToken_);
  fillValueMap(ev, muons, assocOrigTrkRaw, assocOrigTrkToken_);
}

// Define this as a plugin
#include <FWCore/Framework/interface/MakerMacros.h>
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"

typedef MuonExtenderWithMTDT<reco::MuonCollection> MuonExtenderWithMTD;
typedef MuonExtenderWithMTDT<reco::RecoChargedCandidateCollection> MuonChgCandExtenderWithMTD;

DEFINE_FWK_MODULE(MuonExtenderWithMTD);
DEFINE_FWK_MODULE(MuonChgCandExtenderWithMTD);
