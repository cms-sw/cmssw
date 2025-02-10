#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/transform.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "Validation/RecoTrack/interface/trackFromSeedFitFailed.h"

#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

#include "RecoTracker/LST/interface/LSTPixelSeedInput.h"

class LSTPixelSeedInputProducer : public edm::global::EDProducer<> {
public:
  explicit LSTPixelSeedInputProducer(edm::ParameterSet const& iConfig);
  ~LSTPixelSeedInputProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> mfToken_;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  std::vector<edm::EDGetTokenT<edm::View<reco::Track>>> seedTokens_;
  const edm::EDPutTokenT<LSTPixelSeedInput> lstPixelSeedInputPutToken_;
  const edm::EDPutTokenT<TrajectorySeedCollection> lstPixelSeedsPutToken_;
};

LSTPixelSeedInputProducer::LSTPixelSeedInputProducer(edm::ParameterSet const& iConfig)
    : mfToken_(esConsumes()),
      beamSpotToken_(consumes(iConfig.getParameter<edm::InputTag>("beamSpot"))),
      lstPixelSeedInputPutToken_(produces()),
      lstPixelSeedsPutToken_(produces()) {
  seedTokens_ = edm::vector_transform(iConfig.getParameter<std::vector<edm::InputTag>>("seedTracks"),
                                      [&](const edm::InputTag& tag) { return consumes<edm::View<reco::Track>>(tag); });
}

void LSTPixelSeedInputProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));

  desc.add<std::vector<edm::InputTag>>("seedTracks",
                                       std::vector<edm::InputTag>{edm::InputTag("lstInitialStepSeedTracks"),
                                                                  edm::InputTag("lstHighPtTripletStepSeedTracks")});

  descriptions.addWithDefaultLabel(desc);
}

void LSTPixelSeedInputProducer::produce(edm::StreamID iID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // Setup
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
          throw cms::Exception("LSTPixelSeedInputProducer") << "Not pixel hits found!";
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

  LSTPixelSeedInput pixelSeedInput(std::move(see_px),
                                   std::move(see_py),
                                   std::move(see_pz),
                                   std::move(see_dxy),
                                   std::move(see_dz),
                                   std::move(see_ptErr),
                                   std::move(see_etaErr),
                                   std::move(see_stateTrajGlbX),
                                   std::move(see_stateTrajGlbY),
                                   std::move(see_stateTrajGlbZ),
                                   std::move(see_stateTrajGlbPx),
                                   std::move(see_stateTrajGlbPy),
                                   std::move(see_stateTrajGlbPz),
                                   std::move(see_q),
                                   std::move(see_hitIdx));
  iEvent.emplace(lstPixelSeedInputPutToken_, std::move(pixelSeedInput));
  iEvent.emplace(lstPixelSeedsPutToken_, std::move(see_seeds));
}

DEFINE_FWK_MODULE(LSTPixelSeedInputProducer);
