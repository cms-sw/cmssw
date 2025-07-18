#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/TrackReco/interface/DeDxHitInfo.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

//
// class declaration
//

class DeDxEstimatorRekeyer : public edm::global::EDProducer<> {
public:
  explicit DeDxEstimatorRekeyer(const edm::ParameterSet&);
  ~DeDxEstimatorRekeyer() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  template <typename T>
  std::map<std::string, edm::EDGetTokenT<T>> getTokens(const std::vector<edm::InputTag>& tags) {
    std::map<std::string, edm::EDGetTokenT<T>> tokens;
    for (const auto& tag : tags)
      tokens.emplace(tag.label(), consumes<T>(tag));
    return tokens;
  };

  // ----------member data ---------------------------
  const edm::EDGetTokenT<reco::TrackCollection> tracksToken_;
  const edm::EDGetTokenT<reco::DeDxHitInfoAss> dedxHitAssToken_;
  const edm::EDGetTokenT<edm::ValueMap<std::vector<float>>> dedxHitMomToken_;
  const std::map<std::string, edm::EDGetTokenT<edm::ValueMap<reco::DeDxData>>> dedxEstimatorsTokens_;
  const std::map<std::string, edm::EDGetTokenT<pat::PackedCandidateCollection>> packedCandidatesTokens_;
  const std::map<std::string, edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection>>> trk2pcTokens_;
};

void DeDxEstimatorRekeyer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("tracks", {"generalTracks"});
  desc.add<edm::InputTag>("dedxHits", {"dedxHitInfo"});
  desc.add<edm::InputTag>("dedxMomentum", {"dedxHitInfo:momentumAtHit"});
  desc.add<std::vector<edm::InputTag>>(
      "packedCandidates",
      {edm::InputTag("packedPFCandidates"), edm::InputTag("lostTracks"), edm::InputTag("lostTracks:eleTracks")});
  desc.add<std::vector<edm::InputTag>>("dedxEstimators",
                                       {edm::InputTag("dedxHarmonic2"), edm::InputTag("dedxPixelHarmonic2")});
  descriptions.addWithDefaultLabel(desc);
}

DeDxEstimatorRekeyer::DeDxEstimatorRekeyer(const edm::ParameterSet& iConfig)
    : tracksToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"))),
      dedxHitAssToken_(consumes<reco::DeDxHitInfoAss>(iConfig.getParameter<edm::InputTag>("dedxHits"))),
      dedxHitMomToken_(
          consumes<edm::ValueMap<std::vector<float>>>(iConfig.getParameter<edm::InputTag>("dedxMomentum"))),
      dedxEstimatorsTokens_(
          getTokens<edm::ValueMap<reco::DeDxData>>(iConfig.getParameter<std::vector<edm::InputTag>>("dedxEstimators"))),
      packedCandidatesTokens_(getTokens<pat::PackedCandidateCollection>(
          iConfig.getParameter<std::vector<edm::InputTag>>("packedCandidates"))),
      trk2pcTokens_(getTokens<edm::Association<pat::PackedCandidateCollection>>(
          iConfig.getParameter<std::vector<edm::InputTag>>("packedCandidates"))) {
  for (const auto& d : dedxEstimatorsTokens_)
    produces<edm::ValueMap<reco::DeDxData>>(d.first);
  produces<reco::DeDxHitInfoCollection>();
  produces<reco::DeDxHitInfoAss>();
  produces<edm::ValueMap<std::vector<float>>>("momentumAtHit");
}

void DeDxEstimatorRekeyer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // Get input collections
  const auto& tracks = iEvent.getHandle(tracksToken_);
  std::vector<reco::TrackRef> trackRefs;
  trackRefs.reserve(tracks->size());
  for (auto track = tracks->begin(); track != tracks->end(); track++)
    trackRefs.emplace_back(tracks, track - tracks->begin());

  typedef std::map<pat::PackedCandidateRef, reco::TrackRef> PCTrkMap;
  std::vector<std::pair<edm::Handle<pat::PackedCandidateCollection>, PCTrkMap>> pcTrkMap;
  pcTrkMap.reserve(packedCandidatesTokens_.size());
  for (const auto& p : packedCandidatesTokens_) {
    PCTrkMap map;
    const auto& trk2pc = iEvent.get(trk2pcTokens_.at(p.first));
    for (const auto& track : trackRefs) {
      const auto& pc = trk2pc[track];
      if (pc.isNonnull())
        map.emplace(pc, track);
    }
    pcTrkMap.emplace_back(iEvent.getHandle(p.second), map);
  }

  // Rekey dEdx estimators
  for (const auto& d : dedxEstimatorsTokens_) {
    const auto& dedxEstimators = iEvent.get(d.second);
    auto trackDeDxValueMap = std::make_unique<edm::ValueMap<reco::DeDxData>>();
    edm::ValueMap<reco::DeDxData>::Filler filler(*trackDeDxValueMap);
    // Loop over packed candidates
    for (const auto& h : pcTrkMap) {
      std::vector<reco::DeDxData> dedxEstimate(h.first->size());
      for (const auto& p : h.second)
        dedxEstimate[p.first.key()] = dedxEstimators[p.second];
      filler.insert(h.first, dedxEstimate.begin(), dedxEstimate.end());
    }
    // Fill the value map and put it into the event
    filler.fill();
    iEvent.put(std::move(trackDeDxValueMap), d.first);
  }

  // Rekey dEdx hit info
  const auto& dedxHitMom = iEvent.get(dedxHitMomToken_);
  const auto& dedxHitAss = iEvent.get(dedxHitAssToken_);
  const auto& dedxHitInfoHandle = iEvent.getRefBeforePut<reco::DeDxHitInfoCollection>();
  auto dedxHitInfoAssociation = std::make_unique<reco::DeDxHitInfoAss>(dedxHitInfoHandle);
  reco::DeDxHitInfoAss::Filler filler(*dedxHitInfoAssociation);
  auto resultdedxHitColl = std::make_unique<reco::DeDxHitInfoCollection>();
  resultdedxHitColl->reserve(!pcTrkMap.empty() ? pcTrkMap.size() * pcTrkMap[0].second.size() : 0);
  std::vector<std::vector<float>> momenta;
  momenta.reserve(resultdedxHitColl->capacity());
  // Loop over packed candidates
  for (const auto& h : pcTrkMap) {
    std::vector<int> indices(h.first->size(), -1);
    for (const auto& p : h.second) {
      const auto& dedxHit = dedxHitAss[p.second];
      if (dedxHit.isNull())
        continue;
      indices[p.first.key()] = resultdedxHitColl->size();
      resultdedxHitColl->emplace_back(*dedxHit);
      momenta.emplace_back(dedxHitMom[dedxHit]);
    }
    filler.insert(h.first, indices.begin(), indices.end());
  }
  const auto& dedxHitCollHandle = iEvent.put(std::move(resultdedxHitColl));
  // Fill the association map and put it into the event
  filler.fill();
  iEvent.put(std::move(dedxHitInfoAssociation));
  // Fill the value map and put it into the event
  auto dedxMomenta = std::make_unique<edm::ValueMap<std::vector<float>>>();
  edm::ValueMap<std::vector<float>>::Filler mfiller(*dedxMomenta);
  mfiller.insert(dedxHitCollHandle, momenta.begin(), momenta.end());
  mfiller.fill();
  iEvent.put(std::move(dedxMomenta), "momentumAtHit");
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DeDxEstimatorRekeyer);
