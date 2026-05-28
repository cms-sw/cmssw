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
#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/LSTCore/interface/LSTInputHostCollection.h"
#include "RecoTracker/LSTCore/interface/TrackCandidatesHostCollection.h"
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

  const edm::EDGetTokenT<lst::TrackCandidatesBaseHostCollection> lstOutputToken_;
  const edm::EDGetTokenT<lst::LSTInputHostCollection> lstInputToken_;
  const edm::EDGetTokenT<TrajectorySeedCollection> lstPixelSeedToken_;
  const bool includeT5s_;
  const bool includeNonpLSTSs_;
  const bool produceSeeds_;
  const bool produceTrackCandidates_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> mfToken_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorAlongToken_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorOppositeToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tGeomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  std::unique_ptr<SeedCreator> seedCreator_;
  const edm::EDPutTokenT<TrajectorySeedCollection> trajectorySeedPutToken_;
  const edm::EDPutTokenT<TrajectorySeedCollection> trajectorySeedpLSPutToken_;
  const edm::EDPutTokenT<TrackCandidateCollection> trackCandidatePutToken_;
  const edm::EDPutTokenT<TrackCandidateCollection> trackCandidatepTCPutToken_;
  const edm::EDPutTokenT<TrackCandidateCollection> trackCandidateT4T5TCPutToken_;
  const edm::EDPutTokenT<TrackCandidateCollection> trackCandidateNopLSTCPutToken_;
  const edm::EDPutTokenT<TrackCandidateCollection> trackCandidatepTTCPutToken_;
  const edm::EDPutTokenT<TrackCandidateCollection> trackCandidatepLSTCPutToken_;
  const edm::EDPutTokenT<std::vector<SeedStopInfo>> seedStopInfoPutToken_;
  const edm::EDPutTokenT<std::vector<SeedStopInfo>> pTCsSeedStopInfoPutToken_;
  const edm::EDPutTokenT<std::vector<SeedStopInfo>> t4t5TCsSeedStopInfoPutToken_;
  const edm::EDPutTokenT<std::vector<SeedStopInfo>> pTTCsSeedStopInfoPutToken_;
};

LSTOutputConverter::LSTOutputConverter(edm::ParameterSet const& iConfig)
    : lstOutputToken_(consumes(iConfig.getParameter<edm::InputTag>("lstOutput"))),
      lstInputToken_{consumes(iConfig.getParameter<edm::InputTag>("lstInput"))},
      lstPixelSeedToken_{consumes(iConfig.getParameter<edm::InputTag>("lstPixelSeeds"))},
      includeT5s_(iConfig.getParameter<bool>("includeT5s")),
      includeNonpLSTSs_(iConfig.getParameter<bool>("includeNonpLSTSs")),
      produceSeeds_(iConfig.getParameter<bool>("produceSeeds")),
      produceTrackCandidates_(iConfig.getParameter<bool>("produceTrackCandidates")),
      mfToken_(esConsumes()),
      propagatorAlongToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("propagatorAlong"))},
      propagatorOppositeToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("propagatorOpposite"))},
      tGeomToken_(esConsumes()),
      tTopoToken_(esConsumes()),
      seedCreator_(SeedCreatorFactory::get()->create("SeedFromConsecutiveHitsCreator",
                                                     iConfig.getParameter<edm::ParameterSet>("SeedCreatorPSet"),
                                                     consumesCollector())),
      trajectorySeedPutToken_(produces<TrajectorySeedCollection>("")),
      trajectorySeedpLSPutToken_(produceSeeds_ ? produces<TrajectorySeedCollection>("pLSTSsLST")
                                               : edm::EDPutTokenT<TrajectorySeedCollection>{}),
      trackCandidatePutToken_(produceTrackCandidates_ ? produces<TrackCandidateCollection>("")
                                                      : edm::EDPutTokenT<TrackCandidateCollection>{}),
      trackCandidatepTCPutToken_(produceTrackCandidates_ ? produces<TrackCandidateCollection>("pTCsLST")
                                                         : edm::EDPutTokenT<TrackCandidateCollection>{}),
      trackCandidateT4T5TCPutToken_(produceTrackCandidates_ ? produces<TrackCandidateCollection>("t4t5TCsLST")
                                                            : edm::EDPutTokenT<TrackCandidateCollection>{}),
      trackCandidateNopLSTCPutToken_(produceTrackCandidates_ ? produces<TrackCandidateCollection>("nopLSTCsLST")
                                                             : edm::EDPutTokenT<TrackCandidateCollection>{}),
      trackCandidatepTTCPutToken_(produceTrackCandidates_ ? produces<TrackCandidateCollection>("pTTCsLST")
                                                          : edm::EDPutTokenT<TrackCandidateCollection>{}),
      trackCandidatepLSTCPutToken_(produceTrackCandidates_ ? produces<TrackCandidateCollection>("pLSTCsLST")
                                                           : edm::EDPutTokenT<TrackCandidateCollection>{}),
      seedStopInfoPutToken_(produceTrackCandidates_ ? produces<std::vector<SeedStopInfo>>("")
                                                    : edm::EDPutTokenT<std::vector<SeedStopInfo>>{}),
      pTCsSeedStopInfoPutToken_(produceTrackCandidates_ ? produces<std::vector<SeedStopInfo>>("pTCsLST")
                                                        : edm::EDPutTokenT<std::vector<SeedStopInfo>>{}),
      t4t5TCsSeedStopInfoPutToken_(produceTrackCandidates_ ? produces<std::vector<SeedStopInfo>>("t4t5TCsLST")
                                                           : edm::EDPutTokenT<std::vector<SeedStopInfo>>{}),
      pTTCsSeedStopInfoPutToken_(produceTrackCandidates_ ? produces<std::vector<SeedStopInfo>>("pTTCsLST")
                                                         : edm::EDPutTokenT<std::vector<SeedStopInfo>>{}) {}

void LSTOutputConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("lstOutput", edm::InputTag("lstProducer"));
  desc.add<edm::InputTag>("lstInput", edm::InputTag("lstInputProducer"));
  desc.add<edm::InputTag>("lstPixelSeeds", edm::InputTag("lstInputProducer"));
  desc.add<bool>("includeT5s", true);
  desc.add<bool>("includeNonpLSTSs", false);
  desc.add<bool>("produceSeeds", true);
  desc.add<bool>("produceTrackCandidates", true);
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
  auto const& pixelSeedsRBP = edm::RefToBaseProd<TrajectorySeed>(iEvent.getHandle(lstPixelSeedToken_));
  auto const& mf = iSetup.getData(mfToken_);
  auto const& propAlo = iSetup.getData(propagatorAlongToken_);
  auto const& propOppo = iSetup.getData(propagatorOppositeToken_);
  auto const& tracker = iSetup.getData(tGeomToken_);
  const TrackerTopology& tTopo = iSetup.getData(tTopoToken_);

  auto lstOutput_view = lstOutput.const_view();
  unsigned int nTrackCandidates = lstOutput_view.nTrackCandidates();

  auto const outputTSRP = iEvent.getRefBeforePut(trajectorySeedPutToken_);

  TrajectorySeedCollection outputTS, outputpLSTS;
  outputTS.reserve(nTrackCandidates);
  outputpLSTS.reserve(nTrackCandidates);
  TrackCandidateCollection outputTC, outputpTC, outputT4T5TC, outputNopLSTC, outputpTTC, outputpLSTC;
  outputTC.reserve(nTrackCandidates);
  outputpTC.reserve(nTrackCandidates);
  outputT4T5TC.reserve(nTrackCandidates);
  outputNopLSTC.reserve(nTrackCandidates);
  outputpTTC.reserve(nTrackCandidates);
  outputpLSTC.reserve(nTrackCandidates);

  auto OTHits = lstInputHC.const_view().hits().hits();

  TrajectorySeedCollection seeds;
  using Hit = SeedingHitSet::ConstRecHitPointer;
  std::vector<Hit> hitsForSeed;

  LogDebug("LSTOutputConverter") << "nTrackCandidates " << nTrackCandidates;
  for (unsigned int i = 0; i < nTrackCandidates; i++) {
    auto iType = lstOutput_view.trackCandidateType()[i];
    bool const isT5orT4 = (iType == lst::LSTObjType::T5 || iType == lst::LSTObjType::T4);
    LogDebug("LSTOutputConverter") << " cand " << i << " " << iType << " " << lstOutput_view.pixelSeedIndex()[i];
    TrajectorySeed seed;
    edm::RefToBase<TrajectorySeed> seedRef;
    if (!isT5orT4) {
      seed = pixelSeeds[lstOutput_view.pixelSeedIndex()[i]];
      seedRef = {pixelSeedsRBP, lstOutput_view.pixelSeedIndex()[i]};
    }

    edm::OwnVector<TrackingRecHit> recHits;
    if (!isT5orT4) {
      for (auto const& hit : seed.recHits())
        recHits.push_back(hit.clone());
    }

    if (iType != lst::LSTObjType::pLS) {
      // The pixel hits are packed into first kPixelLayerSlots layer slots.
      for (unsigned int layerSlot = lst::Params_TC::kPixelLayerSlots; layerSlot < lst::Params_TC::kLayers;
           ++layerSlot) {
        for (unsigned int hitSlot = 0; hitSlot < lst::Params_TC::kHitsPerLayer; ++hitSlot) {
          unsigned int hitIdx = lstOutput_view.hitIndices()[i][layerSlot][hitSlot];
          if (hitIdx == lst::kTCEmptyHitIdx)
            continue;
          recHits.push_back(OTHits[hitIdx]->clone());
        }
      }

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
    }

    if (iType != lst::LSTObjType::pLS) {
      // For T5/T4: makeSeed is needed whenever seeds or TCs are produced, since the resulting
      // seed is the only source of initial state for T5/T4 track candidates.
      // For other pT objects: makeSeed is only needed for seed output.
      if ((isT5orT4 && (produceSeeds_ || produceTrackCandidates_)) ||
          (!isT5orT4 && produceSeeds_ && includeNonpLSTSs_)) {
        hitsForSeed.clear();
        hitsForSeed.reserve(recHits.size());
        int n = 0;
        unsigned int firstLayer;
        for (auto const& hit : recHits) {
          if (iType == lst::LSTObjType::T5) {
            auto hType = tracker.getDetectorType(hit.geographicalId());
            if (hType != TrackerGeometry::ModuleType::Ph2PSP && n < 2)
              continue;  // the first two should be P
          }
          if (iType == lst::LSTObjType::T4) {
            unsigned int hitLayer = tTopo.layer(hit.geographicalId());
            auto hType = tracker.getDetectorType(hit.geographicalId());
            if (n == 0)
              firstLayer = hitLayer;
            else {
              if (hType == TrackerGeometry::ModuleType::Ph2PSS && hitLayer == firstLayer)
                continue;
            }
          }
          hitsForSeed.emplace_back(dynamic_cast<Hit>(&hit));
          n++;
        }
        GlobalTrackingRegion region;
        seedCreator_->init(region, iSetup, nullptr);
        seeds.clear();
        seedCreator_->makeSeed(seeds, hitsForSeed);
        if (seeds.empty()) {
          edm::LogInfo("LSTOutputConverter") << "failed to convert a LST object to a seed" << i << " " << iType << " "
                                             << lstOutput_view.pixelSeedIndex()[i];
          if (isT5orT4)
            continue;
        }
        if (isT5orT4) {
          seed = seeds[0];
          seedRef = edm::RefToBase<TrajectorySeed>(edm::Ref(outputTSRP, outputTS.size()));
        }

        auto trajectorySeed = (seeds.empty() ? seed : seeds[0]);
        outputTS.emplace_back(trajectorySeed);
        auto const& ss = trajectorySeed.startingState();
        LogDebug("LSTOutputConverter") << "Created a seed with " << trajectorySeed.nHits() << " " << ss.detId() << " "
                                       << ss.pt() << " " << ss.parameters().vector() << " " << ss.error(0);
      }
    } else {
      if (produceSeeds_) {
        outputTS.emplace_back(seed);
        outputpLSTS.emplace_back(seed);
      }
    }

    if (!produceTrackCandidates_)
      continue;

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

      if (!includeT5s_ && isT5orT4)
        continue;

      auto tc = TrackCandidate(recHits, seed, st, seedRef);
      outputTC.emplace_back(tc);
      if (isT5orT4) {
        outputT4T5TC.emplace_back(tc);
        outputNopLSTC.emplace_back(tc);
      } else {
        outputpTC.emplace_back(tc);
        if (iType != lst::LSTObjType::pLS) {
          outputNopLSTC.emplace_back(tc);
          outputpTTC.emplace_back(tc);
        } else {
          outputpLSTC.emplace_back(tc);
        }
      }
    } else {
      edm::LogInfo("LSTOutputConverter") << "Failed to make a candidate initial state. Seed state is " << tsos
                                         << " TC cand " << i << " " << lstOutput_view.pixelSeedIndex()[i] << " "
                                         << lstOutput_view.pixelSeedIndex()[i] << " first hit "
                                         << recHits.front().globalPosition() << " last hit "
                                         << recHits.back().globalPosition();
    }
  }

  LogDebug("LSTOutputConverter") << "done with conversion: Track candidate output size = " << outputpTC.size()
                                 << " (p* objects) + " << outputT4T5TC.size() << " (T5 objects)";

  iEvent.emplace(trajectorySeedPutToken_, std::move(outputTS));
  if (produceSeeds_)
    iEvent.emplace(trajectorySeedpLSPutToken_, std::move(outputpLSTS));
  if (produceTrackCandidates_) {
    //dummy (for now) stop infos: one per used kind of candidates
    iEvent.emplace(seedStopInfoPutToken_, std::vector<SeedStopInfo>(pixelSeeds.size()));
    iEvent.emplace(pTCsSeedStopInfoPutToken_, std::vector<SeedStopInfo>(pixelSeeds.size()));
    iEvent.emplace(t4t5TCsSeedStopInfoPutToken_, std::vector<SeedStopInfo>(outputT4T5TC.size()));
    iEvent.emplace(pTTCsSeedStopInfoPutToken_, std::vector<SeedStopInfo>(pixelSeeds.size()));
    iEvent.emplace(trackCandidatePutToken_, std::move(outputTC));
    iEvent.emplace(trackCandidatepTCPutToken_, std::move(outputpTC));
    iEvent.emplace(trackCandidateT4T5TCPutToken_, std::move(outputT4T5TC));
    iEvent.emplace(trackCandidateNopLSTCPutToken_, std::move(outputNopLSTC));
    iEvent.emplace(trackCandidatepTTCPutToken_, std::move(outputpTTC));
    iEvent.emplace(trackCandidatepLSTCPutToken_, std::move(outputpLSTC));
  }
}

DEFINE_FWK_MODULE(LSTOutputConverter);
