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

#include "FWCore/Utilities/interface/transform.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "Validation/RecoTrack/interface/trackFromSeedFitFailed.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

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
    std::vector<edm::EDGetTokenT<edm::View<reco::Track>>> seedTokens_;
    const edm::EDPutTokenT<TrajectorySeedCollection> lstPixelSeedsPutToken_;

    const edm::EDPutTokenT<lst::LSTInputHostCollection> lstInputPutToken_;
  };

  LSTInputProducer::LSTInputProducer(edm::ParameterSet const& iConfig)
      : EDProducer<>(iConfig),
        ptCut_(iConfig.getParameter<double>("ptCut")),
        phase2OTRecHitToken_(consumes(iConfig.getParameter<edm::InputTag>("phase2OTRecHits"))),
        mfToken_(esConsumes()),
        beamSpotToken_(consumes(iConfig.getParameter<edm::InputTag>("beamSpot"))),
        lstPixelSeedsPutToken_(produces()),
        lstInputPutToken_(produces()) {
    seedTokens_ =
        edm::vector_transform(iConfig.getParameter<std::vector<edm::InputTag>>("seedTracks"),
                              [&](const edm::InputTag& tag) { return consumes<edm::View<reco::Track>>(tag); });
  }

  void LSTInputProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<double>("ptCut", 0.8);

    desc.add<edm::InputTag>("phase2OTRecHits", edm::InputTag("siPhase2RecHits"));

    desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
    desc.add<std::vector<edm::InputTag>>("seedTracks",
                                         std::vector<edm::InputTag>{edm::InputTag("lstInitialStepSeedTracks"),
                                                                    edm::InputTag("lstHighPtTripletStepSeedTracks")});

    descriptions.addWithDefaultLabel(desc);
  }

  void LSTInputProducer::produce(edm::StreamID iID, device::Event& iEvent, const device::EventSetup& iSetup) const {
    // Get the phase2OTRecHits
    auto const& phase2OTHits = iEvent.get(phase2OTRecHitToken_);

    std::vector<unsigned int> ph2_detId;
    ph2_detId.reserve(phase2OTHits.dataSize());
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
        ph2_x.push_back(hit.globalPosition().x());
        ph2_y.push_back(hit.globalPosition().y());
        ph2_z.push_back(hit.globalPosition().z());
        ph2_hits.push_back(&hit);
      }
    }

    // Get the pixel seeds
    auto const& mf = iSetup.getData(mfToken_);
    auto const& bs = iEvent.get(beamSpotToken_);

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
    TrajectorySeedCollection see_seeds;

    for (size_t iColl = 0; iColl < seedTokens_.size(); ++iColl) {
      // Get seed tokens
      auto const& seedToken = seedTokens_[iColl];
      auto const& seedTracks = iEvent.get(seedToken);

      if (seedTracks.empty())
        continue;

      // Get seed track refs
      edm::RefToBaseVector<reco::Track> seedTrackRefs;
      for (edm::View<reco::Track>::size_type i = 0; i < seedTracks.size(); ++i) {
        seedTrackRefs.push_back(seedTracks.refAt(i));
      }

      edm::ProductID id = seedTracks[0].seedRef().id();

      for (size_t iSeed = 0; iSeed < seedTrackRefs.size(); ++iSeed) {
        auto const& seedTrackRef = seedTrackRefs[iSeed];
        auto const& seedTrack = *seedTrackRef;
        auto const& seedRef = seedTrack.seedRef();
        auto const& seed = *seedRef;

        if (seedRef.id() != id)
          throw cms::Exception("LogicError")
              << "All tracks in 'TracksFromSeeds' collection should point to seeds in the same collection. Now the "
                 "element 0 had ProductID "
              << id << " while the element " << seedTrackRef.key() << " had " << seedTrackRef.id() << ".";

        const bool seedFitOk = !trackFromSeedFitFailed(seedTrack);

        const TrackingRecHit* lastRecHit = &*(seed.recHits().end() - 1);
        TrajectoryStateOnSurface tsos =
            trajectoryStateTransform::transientState(seed.startingState(), lastRecHit->surface(), &mf);
        auto const& stateGlobal = tsos.globalParameters();

        std::vector<int> hitIdx;
        for (auto const& hit : seed.recHits()) {
          int subid = hit.geographicalId().subdetId();
          if (subid == (int)PixelSubdetector::PixelBarrel || subid == (int)PixelSubdetector::PixelEndcap) {
            const BaseTrackerRecHit* bhit = dynamic_cast<const BaseTrackerRecHit*>(&hit);
            const auto& clusterRef = bhit->firstClusterRef();
            const auto clusterKey = clusterRef.cluster_pixel().key();
            hitIdx.push_back(clusterKey);
          } else {
            throw cms::Exception("LSTInputProducer") << "Not pixel hits found!";
          }
        }

        // Fill output
        see_px.push_back(seedFitOk ? seedTrack.px() : 0);
        see_py.push_back(seedFitOk ? seedTrack.py() : 0);
        see_pz.push_back(seedFitOk ? seedTrack.pz() : 0);
        see_dxy.push_back(seedFitOk ? seedTrack.dxy(bs.position()) : 0);
        see_dz.push_back(seedFitOk ? seedTrack.dz(bs.position()) : 0);
        see_ptErr.push_back(seedFitOk ? seedTrack.ptError() : 0);
        see_etaErr.push_back(seedFitOk ? seedTrack.etaError() : 0);
        see_stateTrajGlbX.push_back(stateGlobal.position().x());
        see_stateTrajGlbY.push_back(stateGlobal.position().y());
        see_stateTrajGlbZ.push_back(stateGlobal.position().z());
        see_stateTrajGlbPx.push_back(stateGlobal.momentum().x());
        see_stateTrajGlbPy.push_back(stateGlobal.momentum().y());
        see_stateTrajGlbPz.push_back(stateGlobal.momentum().z());
        see_q.push_back(seedTrack.charge());
        see_hitIdx.push_back(hitIdx);
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
                                        {},
                                        ph2_detId,
                                        ph2_x,
                                        ph2_y,
                                        ph2_z,
                                        ph2_hits,
                                        ptCut_);

    iEvent.emplace(lstInputPutToken_, std::move(lstInputHC));
    iEvent.emplace(lstPixelSeedsPutToken_, std::move(see_seeds));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(LSTInputProducer);
