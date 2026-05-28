#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "FWCore/Utilities/interface/transform.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/TrajectoryState/interface/PerigeeConversions.h"

#include "RecoTracker/LSTCore/interface/LSTInputHostCollection.h"
#include "RecoTracker/LSTCore/interface/LSTPrepareInput.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class LSTInputProducer : public global::EDProducer<> {
  public:
    LSTInputProducer(edm::ParameterSet const& iConfig);
    ~LSTInputProducer() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void produce(edm::StreamID, device::Event& iEvent, const device::EventSetup& iSetup) const override;

    const double ptCut_;

    const edm::EDGetTokenT<Phase2TrackerRecHit1DCollectionNew> phase2OTRecHitToken_;

    const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> mfToken_;
    const edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
    const std::vector<edm::EDGetTokenT<TrajectorySeedCollection>> seedTokens_;
    const edm::EDPutTokenT<TrajectorySeedCollection> lstPixelSeedsPutToken_;

    const edm::EDPutTokenT<lst::LSTInputHostCollection> lstInputPutToken_;
  };

  LSTInputProducer::LSTInputProducer(edm::ParameterSet const& iConfig)
      : EDProducer<>(iConfig),
        ptCut_(iConfig.getParameter<double>("ptCut")),
        phase2OTRecHitToken_(consumes(iConfig.getParameter<edm::InputTag>("phase2OTRecHits"))),
        mfToken_(esConsumes()),
        beamSpotToken_(consumes(iConfig.getParameter<edm::InputTag>("beamSpot"))),
        seedTokens_(
            edm::vector_transform(iConfig.getParameter<std::vector<edm::InputTag>>("pixelSeeds"),
                                  [&](const edm::InputTag& tag) { return consumes<TrajectorySeedCollection>(tag); })),
        lstPixelSeedsPutToken_(produces()),
        lstInputPutToken_(produces()) {}

  void LSTInputProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<double>("ptCut", 0.8);

    desc.add<edm::InputTag>("phase2OTRecHits", edm::InputTag("siPhase2RecHits"));

    desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
    desc.add<std::vector<edm::InputTag>>(
        "pixelSeeds",
        std::vector<edm::InputTag>{edm::InputTag("initialStepSeeds"), edm::InputTag("highPtTripletStepSeeds")});

    descriptions.addWithDefaultLabel(desc);
  }

  void LSTInputProducer::produce(edm::StreamID iID, device::Event& iEvent, const device::EventSetup& iSetup) const {
    // Get the phase2OTRecHits
    auto const& phase2OTHits = iEvent.get(phase2OTRecHitToken_);

    std::vector<unsigned int> ph2_detId;
    ph2_detId.reserve(phase2OTHits.dataSize());
    std::vector<uint16_t> ph2_clustSize;
    ph2_clustSize.reserve(phase2OTHits.dataSize());
    std::vector<float> ph2_x;
    ph2_x.reserve(phase2OTHits.dataSize());
    std::vector<float> ph2_y;
    ph2_y.reserve(phase2OTHits.dataSize());
    std::vector<float> ph2_z;
    ph2_z.reserve(phase2OTHits.dataSize());
    std::vector<TrackingRecHit const*> ph2_hits;
    ph2_hits.reserve(phase2OTHits.dataSize());

    for (auto const& it : phase2OTHits) {
      const DetId hitId = it.detId();
      for (auto const& hit : it) {
        ph2_detId.push_back(hitId.rawId());
        ph2_clustSize.push_back(hit.cluster()->size());
        ph2_x.push_back(hit.globalPosition().x());
        ph2_y.push_back(hit.globalPosition().y());
        ph2_z.push_back(hit.globalPosition().z());
        ph2_hits.push_back(&hit);
      }
    }

    // Get the pixel seeds
    auto const& mf = iSetup.getData(mfToken_);
    auto const& bs = iEvent.get(beamSpotToken_);

    TSCBLBuilderNoMaterial tscblBuilder;

    // Vector definitions
    std::vector<float> see_px;
    std::vector<float> see_py;
    std::vector<float> see_pz;
    std::vector<float> see_dxy;
    std::vector<float> see_dz;
    std::vector<float> see_ptErr;
    std::vector<float> see_etaErr;
    std::vector<float> see_stateTrajGlbX;
    std::vector<float> see_stateTrajGlbY;
    std::vector<float> see_stateTrajGlbZ;
    std::vector<float> see_stateTrajGlbPx;
    std::vector<float> see_stateTrajGlbPy;
    std::vector<float> see_stateTrajGlbPz;
    std::vector<int> see_q;
    std::vector<std::vector<int>> see_hitIdx;
    std::vector<std::vector<int>> see_hitType;
    TrajectorySeedCollection see_seeds;

    for (auto const& seedToken : seedTokens_) {
      auto const& seeds = iEvent.get(seedToken);

      if (seeds.empty())
        continue;

      for (auto const& seed : seeds) {
        const TrackingRecHit* lastRecHit = &*(seed.recHits().end() - 1);
        TrajectoryStateOnSurface tsos =
            trajectoryStateTransform::transientState(seed.startingState(), lastRecHit->surface(), &mf);
        auto const& stateGlobal = tsos.globalParameters();

        // Propagate to beam line to get perigee parameters (replicates TrackFromSeedProducer)
        TrajectoryStateClosestToBeamLine tscbl = tscblBuilder(*(tsos.freeState()), bs);
        const bool tscblValid = tscbl.isValid();

        float px = 0, py = 0, pz = 0, dxy = 0, dz = 0, ptErr = 0, etaErr = 0;
        int charge = tscblValid ? tsos.charge() : 0;

        if (tscblValid) {
          auto const& fts = tscbl.trackStateAtPCA();
          auto const& mom = fts.momentum();
          auto const& pos = fts.position();
          auto const& bsPos = bs.position();
          px = mom.x();
          py = mom.y();
          pz = mom.z();
          dxy = (-(pos.x() - bsPos.x()) * mom.y() + (pos.y() - bsPos.y()) * mom.x()) / mom.perp();
          dz = (pos.z() - bsPos.z()) - ((pos.x() - bsPos.x()) * mom.x() + (pos.y() - bsPos.y()) * mom.y()) /
                                           mom.perp() * (mom.z() / mom.perp());

          // Compute ptErr and etaErr replicating reco::TrackBase::ptError() and etaError()
          PerigeeTrajectoryError periErr = PerigeeConversions::ftsToPerigeeError(fts);
          auto const& errMat = periErr.covarianceMatrix();
          double pt = mom.perp();
          double p = mom.mag();
          double pz = mom.z();
          double q = static_cast<double>(charge);
          // Full error propagation matching reco::TrackBase::ptError2():
          //   pt2*p2/q2 * cov(qoverp,qoverp) + 2*sqrt(p2*pt2)/q * pz * cov(qoverp,lambda) + pz2 * cov(lambda,lambda)
          double pt2 = pt * pt;
          double p2 = p * p;
          ptErr = std::sqrt(pt2 * p2 / (q * q) * errMat(0, 0) + 2.0 * std::sqrt(p2 * pt2) / q * pz * errMat(0, 1) +
                            pz * pz * errMat(1, 1));
          // etaError() = sqrt(cov(lambda,lambda)) * p/pt
          etaErr = std::sqrt(errMat(1, 1)) * p / pt;
        }

        std::vector<int> hitIdx;
        std::vector<int> hitType;
        for (auto const& hit : seed.recHits()) {
          auto det = hit.geographicalId().det();
          if (det == DetId::Tracker) {
            const BaseTrackerRecHit* bhit = dynamic_cast<const BaseTrackerRecHit*>(&hit);
            const auto& clusterRef = bhit->firstClusterRef();
            hitIdx.push_back(clusterRef.index());
            if (clusterRef.isPixel()) {
              hitType.push_back(static_cast<int>(lst::HitType::Pixel));
            } else if (clusterRef.isPhase2()) {
              hitType.push_back(static_cast<int>(lst::HitType::Phase2OT));
            } else {
              throw cms::Exception("LSTInputProducer") << "Unknown tracker hit type found!";
            }
          } else {
            throw cms::Exception("LSTInputProducer") << "Not tracker hit found!";
          }
        }

        // Fill output
        see_px.push_back(px);
        see_py.push_back(py);
        see_pz.push_back(pz);
        see_dxy.push_back(dxy);
        see_dz.push_back(dz);
        see_ptErr.push_back(ptErr);
        see_etaErr.push_back(etaErr);
        see_stateTrajGlbX.push_back(stateGlobal.position().x());
        see_stateTrajGlbY.push_back(stateGlobal.position().y());
        see_stateTrajGlbZ.push_back(stateGlobal.position().z());
        see_stateTrajGlbPx.push_back(stateGlobal.momentum().x());
        see_stateTrajGlbPy.push_back(stateGlobal.momentum().y());
        see_stateTrajGlbPz.push_back(stateGlobal.momentum().z());
        see_q.push_back(charge);
        see_hitIdx.emplace_back(std::move(hitIdx));
        see_hitType.emplace_back(std::move(hitType));
        see_seeds.push_back(seed);
      }
    }

    auto lstInputHC = lst::prepareInput(see_px,
                                        see_py,
                                        see_pz,
                                        see_dxy,
                                        see_dz,
                                        see_ptErr,
                                        see_etaErr,
                                        see_stateTrajGlbX,
                                        see_stateTrajGlbY,
                                        see_stateTrajGlbZ,
                                        see_stateTrajGlbPx,
                                        see_stateTrajGlbPy,
                                        see_stateTrajGlbPz,
                                        see_q,
                                        see_hitIdx,
                                        see_hitType,
                                        {},
                                        ph2_detId,
                                        ph2_clustSize,
                                        ph2_x,
                                        ph2_y,
                                        ph2_z,
                                        ph2_hits,
                                        ptCut_,
                                        iEvent.queue());

    iEvent.emplace(lstInputPutToken_, std::move(lstInputHC));
    iEvent.emplace(lstPixelSeedsPutToken_, std::move(see_seeds));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(LSTInputProducer);
