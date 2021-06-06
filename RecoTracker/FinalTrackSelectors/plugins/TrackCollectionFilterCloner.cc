#include "RecoTracker/FinalTrackSelectors/interface/TrackCollectionCloner.h"

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"

#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <memory>

using namespace reco;

namespace {
  class TrackCollectionFilterCloner final : public edm::global::EDProducer<> {
  public:
    /// constructor
    explicit TrackCollectionFilterCloner(const edm::ParameterSet& iConfig);
    /// destructor
    ~TrackCollectionFilterCloner() override;

    /// alias for container of candidate and input tracks
    using CandidateToDuplicate = std::vector<std::pair<int, int>>;

    using RecHitContainer = edm::OwnVector<TrackingRecHit>;

    using MVACollection = std::vector<float>;
    using QualityMaskCollection = std::vector<unsigned char>;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    /// produce one event
    void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  private:
    TrackCollectionCloner collectionCloner;
    const TrackCollectionCloner::Tokens originalTrackSource_;

    const edm::EDGetTokenT<MVACollection> originalMVAValsToken_;
    const edm::EDGetTokenT<QualityMaskCollection> originalQualValsToken_;

    const reco::TrackBase::TrackQuality minQuality_;
  };
}  // namespace

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "TrackingTools/PatternTools/interface/ClusterRemovalRefSetter.h"

#include "FWCore/Framework/interface/Event.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

#include <tuple>
#include <array>
#include "CommonTools/Utils/interface/DynArray.h"

namespace {
  void TrackCollectionFilterCloner::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("originalSource", edm::InputTag());
    desc.add<edm::InputTag>("originalMVAVals", edm::InputTag());
    desc.add<edm::InputTag>("originalQualVals", edm::InputTag());
    desc.add<std::string>("minQuality", "loose");
    TrackCollectionCloner::fill(desc);
    descriptions.add("TrackCollectionFilterCloner", desc);
  }

  TrackCollectionFilterCloner::TrackCollectionFilterCloner(const edm::ParameterSet& iConfig)
      : collectionCloner(producesCollector(), iConfig, true),
        originalTrackSource_(iConfig.getParameter<edm::InputTag>("originalSource"), consumesCollector()),
        originalMVAValsToken_(consumes<MVACollection>(iConfig.getParameter<edm::InputTag>("originalMVAVals"))),
        originalQualValsToken_(
            consumes<QualityMaskCollection>(iConfig.getParameter<edm::InputTag>("originalQualVals"))),
        minQuality_(reco::TrackBase::qualityByName(iConfig.getParameter<std::string>("minQuality"))) {
    produces<MVACollection>("MVAValues");
    produces<QualityMaskCollection>("QualityMasks");
  }

  TrackCollectionFilterCloner::~TrackCollectionFilterCloner() {}

  void TrackCollectionFilterCloner::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
    TrackCollectionCloner::Producer producer(iEvent, collectionCloner);

    // load original tracks
    auto const& originalsTracks = originalTrackSource_.tracks(iEvent);
    //  auto const & originalIndices = originalTrackSource_.indicesInput(iEvent);
    auto nTracks = originalsTracks.size();

    edm::Handle<MVACollection> originalMVAStore;
    iEvent.getByToken(originalMVAValsToken_, originalMVAStore);
    assert((*originalMVAStore).size() == nTracks);

    edm::Handle<QualityMaskCollection> originalQualStore;
    iEvent.getByToken(originalQualValsToken_, originalQualStore);
    assert((*originalQualStore).size() == nTracks);

    // define minimum quality as set in the config file
    unsigned char qualMask = ~0;
    if (minQuality_ != reco::TrackBase::undefQuality)
      qualMask = 1 << minQuality_;

    // products
    std::vector<unsigned int> selId;
    auto pmvas = std::make_unique<MVACollection>();
    auto pquals = std::make_unique<QualityMaskCollection>();

    auto k = 0U;
    for (auto j = 0U; j < nTracks; ++j) {
      if (!(qualMask & (*originalQualStore)[j]))
        continue;

      selId.push_back(j);
      pmvas->push_back((*originalMVAStore)[j]);
      pquals->push_back((*originalQualStore)[j]);

      ++k;
    }

    // clone selected tracks...
    auto nsel = k;
    auto isel = 0U;
    assert(producer.selTracks_->size() == isel);
    producer(originalTrackSource_, selId);
    assert(producer.selTracks_->size() == nsel);

    for (; isel < nsel; ++isel) {
      auto& otk = (*producer.selTracks_)[isel];
      otk.setQualityMask((*pquals)[isel]);
    }
    assert(producer.selTracks_->size() == pmvas->size());
    assert(producer.selTracks_->size() == pquals->size());

    iEvent.put(std::move(pmvas), "MVAValues");
    iEvent.put(std::move(pquals), "QualityMasks");
  }

}  // namespace

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TrackCollectionFilterCloner);
