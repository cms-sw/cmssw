#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/SeedStopInfo.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/LSTCore/interface/LSTInputHostCollection.h"
#include "RecoTracker/LST/interface/LSTOutput.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"

#include "RecoTracker/TkSeedGenerator/interface/SeedCreator.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedCreatorFactory.h"

#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

class LSTOutputConverter : public edm::stream::EDProducer<> {
public:
  explicit LSTOutputConverter(edm::ParameterSet const& iConfig);
  ~LSTOutputConverter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  const edm::EDGetTokenT<LSTOutput> lstOutputToken_;
  const edm::EDGetTokenT<lst::LSTInputHostCollection> lstInputToken_;
  const edm::EDGetTokenT<TrajectorySeedCollection> lstPixelSeedToken_;
  const bool includeT5s_;
  const bool includeNonpLSTSs_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> mfToken_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorAlongToken_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorOppositeToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tGeomToken_;
  std::unique_ptr<SeedCreator> seedCreator_;
  const edm::EDPutTokenT<TrajectorySeedCollection> trajectorySeedPutToken_;
  const edm::EDPutTokenT<TrajectorySeedCollection> trajectorySeedpLSPutToken_;
  const edm::EDPutTokenT<TrackCandidateCollection> trackCandidatePutToken_;
  const edm::EDPutTokenT<TrackCandidateCollection> trackCandidatepTCPutToken_;
  const edm::EDPutTokenT<TrackCandidateCollection> trackCandidateT5TCPutToken_;
  const edm::EDPutTokenT<TrackCandidateCollection> trackCandidateNopLSTCPutToken_;
  const edm::EDPutTokenT<TrackCandidateCollection> trackCandidatepTTCPutToken_;
  const edm::EDPutTokenT<TrackCandidateCollection> trackCandidatepLSTCPutToken_;
  const edm::EDPutTokenT<std::vector<SeedStopInfo>> seedStopInfoPutToken_;
};

LSTOutputConverter::LSTOutputConverter(edm::ParameterSet const& iConfig)
    : lstOutputToken_(consumes(iConfig.getParameter<edm::InputTag>("lstOutput"))),
      lstInputToken_{consumes(iConfig.getParameter<edm::InputTag>("lstInput"))},
      lstPixelSeedToken_{consumes(iConfig.getParameter<edm::InputTag>("lstPixelSeeds"))},
      includeT5s_(iConfig.getParameter<bool>("includeT5s")),
      includeNonpLSTSs_(iConfig.getParameter<bool>("includeNonpLSTSs")),
      mfToken_(esConsumes()),
      propagatorAlongToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("propagatorAlong"))},
      propagatorOppositeToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("propagatorOpposite"))},
      tGeomToken_(esConsumes()),
      seedCreator_(SeedCreatorFactory::get()->create("SeedFromConsecutiveHitsCreator",
                                                     iConfig.getParameter<edm::ParameterSet>("SeedCreatorPSet"),
                                                     consumesCollector())),
      // FIXME: need to make creation configurable:
      // - A toggle to not produce TSs at all could be useful to save memory;
      //   it won't affect speed though
      // - The minimal set for TCs is t5TCsLST, pTTCsLST and pLSTCsLST.
      //   That would complicate the handling of collections though,
      //   so it is deferred to when we have a clearer picture of what's needed.
      trajectorySeedPutToken_(produces("")),
      trajectorySeedpLSPutToken_(produces("pLSTSsLST")),
      trackCandidatePutToken_(produces("")),
      trackCandidatepTCPutToken_(produces("pTCsLST")),
      trackCandidateT5TCPutToken_(produces("t5TCsLST")),
      trackCandidateNopLSTCPutToken_(produces("nopLSTCsLST")),
      trackCandidatepTTCPutToken_(produces("pTTCsLST")),
      trackCandidatepLSTCPutToken_(produces("pLSTCsLST")),
      seedStopInfoPutToken_(produces()) {}

void LSTOutputConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("lstOutput", edm::InputTag("lstProducer"));
  desc.add<edm::InputTag>("lstInput", edm::InputTag("lstInputProducer"));
  desc.add<edm::InputTag>("lstPixelSeeds", edm::InputTag("lstInputProducer"));
  desc.add<bool>("includeT5s", true);
  desc.add<bool>("includeNonpLSTSs", false);
  desc.add("propagatorAlong", edm::ESInputTag{"", "PropagatorWithMaterial"});
  desc.add("propagatorOpposite", edm::ESInputTag{"", "PropagatorWithMaterialOpposite"});

  edm::ParameterSetDescription psd0;
  psd0.add<std::string>("ComponentName", std::string("SeedFromConsecutiveHitsCreator"));
  psd0.add<std::string>("propagator", std::string("PropagatorWithMaterial"));
  psd0.add<double>("SeedMomentumForBOFF", 5.0);
  psd0.add<double>("OriginTransverseErrorMultiplier", 1.0);
  psd0.add<double>("MinOneOverPtError", 1.0);
  psd0.add<std::string>("magneticField", std::string(""));
  psd0.add<std::string>("TTRHBuilder", std::string("WithTrackAngle"));
  psd0.add<bool>("forceKinematicWithRegionDirection", false);
  desc.add<edm::ParameterSetDescription>("SeedCreatorPSet", psd0);

  descriptions.addWithDefaultLabel(desc);
}

void LSTOutputConverter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Setup
  auto const& lstOutput = iEvent.get(lstOutputToken_);
  auto const& lstInputHC = iEvent.get(lstInputToken_);
  auto const& pixelSeeds = iEvent.get(lstPixelSeedToken_);
  auto const& mf = iSetup.getData(mfToken_);
  auto const& propAlo = iSetup.getData(propagatorAlongToken_);
  auto const& propOppo = iSetup.getData(propagatorOppositeToken_);
  auto const& tracker = iSetup.getData(tGeomToken_);

  // Vector definitions
  std::vector<std::vector<unsigned int>> const& lstTC_hitIdx = lstOutput.hitIdx();
  std::vector<unsigned int> const& lstTC_len = lstOutput.len();
  std::vector<int> const& lstTC_seedIdx = lstOutput.seedIdx();
  std::vector<short> const& lstTC_trackCandidateType = lstOutput.trackCandidateType();

  TrajectorySeedCollection outputTS, outputpLSTS;
  outputTS.reserve(lstTC_len.size());
  outputpLSTS.reserve(lstTC_len.size());
  TrackCandidateCollection outputTC, outputpTC, outputT5TC, outputNopLSTC, outputpTTC, outputpLSTC;
  outputTC.reserve(lstTC_len.size());
  outputpTC.reserve(lstTC_len.size());
  outputT5TC.reserve(lstTC_len.size());
  outputNopLSTC.reserve(lstTC_len.size());
  outputpTTC.reserve(lstTC_len.size());
  outputpLSTC.reserve(lstTC_len.size());

  auto OTHits = lstInputHC.const_view<lst::InputHitsSoA>().hits();

  LogDebug("LSTOutputConverter") << "lstTC size " << lstTC_len.size();
  for (unsigned int i = 0; i < lstTC_len.size(); i++) {
    LogDebug("LSTOutputConverter") << " cand " << i << " " << lstTC_len[i] << " " << lstTC_seedIdx[i];
    TrajectorySeed seed;
    if (lstTC_trackCandidateType[i] != LSTOutput::LSTTCType::T5)
      seed = pixelSeeds[lstTC_seedIdx[i]];

    edm::OwnVector<TrackingRecHit> recHits;
    if (lstTC_trackCandidateType[i] != LSTOutput::LSTTCType::T5) {
      for (auto const& hit : seed.recHits())
        recHits.push_back(hit.clone());
    }

    unsigned int const nPixelHits = lstTC_trackCandidateType[i] == LSTOutput::LSTTCType::T5 ? 0 : recHits.size();
    for (unsigned int j = nPixelHits; j < lstTC_hitIdx[i].size(); j++)
      recHits.push_back(OTHits[lstTC_hitIdx[i][j]]->clone());

    recHits.sort([](const auto& a, const auto& b) {
      const auto asub = a.det()->subDetector();
      const auto bsub = b.det()->subDetector();
      if (GeomDetEnumerators::isInnerTracker(asub) && GeomDetEnumerators::isOuterTracker(bsub)) {
        return true;
      } else if (GeomDetEnumerators::isOuterTracker(asub) && GeomDetEnumerators::isInnerTracker(bsub)) {
        return false;
      } else if (asub != bsub) {
        return asub < bsub;
      } else {
        const auto& apos = a.surface();
        const auto& bpos = b.surface();
        if (GeomDetEnumerators::isBarrel(asub)) {
          return apos->rSpan().first < bpos->rSpan().first;
        } else {
          return std::abs(apos->zSpan().first) < std::abs(bpos->zSpan().first);
        }
      }
    });

    TrajectorySeedCollection seeds;
    if (lstTC_trackCandidateType[i] != LSTOutput::LSTTCType::pLS) {
      // Construct a full-length TrajectorySeed always for T5s,
      // only when required by a flag for other pT objects.
      if (includeNonpLSTSs_ || lstTC_trackCandidateType[i] == LSTOutput::LSTTCType::T5) {
        using Hit = SeedingHitSet::ConstRecHitPointer;
        std::vector<Hit> hitsForSeed;
        hitsForSeed.reserve(lstTC_len[i]);
        int nHits = 0;
        for (auto const& hit : recHits) {
          if (lstTC_trackCandidateType[i] == LSTOutput::LSTTCType::T5) {
            auto hType = tracker.getDetectorType(hit.geographicalId());
            if (hType != TrackerGeometry::ModuleType::Ph2PSP && nHits < 2)
              continue;  // the first two should be P
          }
          hitsForSeed.emplace_back(dynamic_cast<Hit>(&hit));
          nHits++;
        }
        GlobalTrackingRegion region;
        seedCreator_->init(region, iSetup, nullptr);
        seedCreator_->makeSeed(seeds, hitsForSeed);
        if (seeds.empty()) {
          edm::LogInfo("LSTOutputConverter")
              << "failed to convert a LST object to a seed" << i << " " << lstTC_len[i] << " " << lstTC_seedIdx[i];
          if (lstTC_trackCandidateType[i] == LSTOutput::LSTTCType::T5)
            continue;
        }
        if (lstTC_trackCandidateType[i] == LSTOutput::LSTTCType::T5)
          seed = seeds[0];

        auto trajectorySeed = (seeds.empty() ? seed : seeds[0]);
        outputTS.emplace_back(trajectorySeed);
        auto const& ss = trajectorySeed.startingState();
        LogDebug("LSTOutputConverter") << "Created a seed with " << seed.nHits() << " " << ss.detId() << " " << ss.pt()
                                       << " " << ss.parameters().vector() << " " << ss.error(0);
      }
    } else {
      outputTS.emplace_back(seed);
      outputpLSTS.emplace_back(seed);
    }

    TrajectoryStateOnSurface tsos =
        trajectoryStateTransform::transientState(seed.startingState(), (seed.recHits().end() - 1)->surface(), &mf);
    tsos.rescaleError(100.);
    auto tsosPair = propOppo.propagateWithPath(tsos, *recHits[0].surface());
    if (!tsosPair.first.isValid()) {
      LogDebug("LSTOutputConverter") << "Propagating to startingState opposite to momentum failed, trying along next";
      tsosPair = propAlo.propagateWithPath(tsos, *recHits[0].surface());
    }
    if (tsosPair.first.isValid()) {
      PTrajectoryStateOnDet st =
          trajectoryStateTransform::persistentState(tsosPair.first, recHits[0].det()->geographicalId().rawId());

      if (lstTC_trackCandidateType[i] == LSTOutput::LSTTCType::T5) {
        if (!includeT5s_) {
          continue;
        } else {
          auto tc = TrackCandidate(recHits, seed, st);
          outputTC.emplace_back(tc);
          outputT5TC.emplace_back(tc);
          outputNopLSTC.emplace_back(tc);
        }
      } else {
        auto tc = TrackCandidate(recHits, seed, st);
        outputTC.emplace_back(tc);
        outputpTC.emplace_back(tc);
        if (lstTC_trackCandidateType[i] != LSTOutput::LSTTCType::pLS) {
          outputNopLSTC.emplace_back(tc);
          outputpTTC.emplace_back(tc);
        } else {
          outputpLSTC.emplace_back(tc);
        }
      }
    } else {
      edm::LogInfo("LSTOutputConverter") << "Failed to make a candidate initial state. Seed state is " << tsos
                                         << " TC cand " << i << " " << lstTC_len[i] << " " << lstTC_seedIdx[i]
                                         << " first hit " << recHits.front().globalPosition() << " last hit "
                                         << recHits.back().globalPosition();
    }
  }

  LogDebug("LSTOutputConverter") << "done with conversion: Track candidate output size = " << outputpTC.size()
                                 << " (p* objects) + " << outputT5TC.size() << " (T5 objects)";
  std::vector<SeedStopInfo> outputSeedStopInfo(pixelSeeds.size());
  iEvent.emplace(trajectorySeedPutToken_, std::move(outputTS));
  iEvent.emplace(trajectorySeedpLSPutToken_, std::move(outputpLSTS));
  iEvent.emplace(trackCandidatePutToken_, std::move(outputTC));
  iEvent.emplace(trackCandidatepTCPutToken_, std::move(outputpTC));
  iEvent.emplace(trackCandidateT5TCPutToken_, std::move(outputT5TC));
  iEvent.emplace(trackCandidateNopLSTCPutToken_, std::move(outputNopLSTC));
  iEvent.emplace(trackCandidatepTTCPutToken_, std::move(outputpTTC));
  iEvent.emplace(trackCandidatepLSTCPutToken_, std::move(outputpLSTC));
  iEvent.emplace(seedStopInfoPutToken_, std::move(outputSeedStopInfo));  //dummy stop info
}

DEFINE_FWK_MODULE(LSTOutputConverter);
