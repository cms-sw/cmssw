#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "RecoEgamma/EgammaIsolationAlgos/interface/SuperclusTkIsolFromCands.h"

// copy-pasted from ElectronHEEPIDValueMapProducer

class SuperclusValueMapProducer : public edm::stream::EDProducer<> {
public:
  explicit SuperclusValueMapProducer(const edm::ParameterSet&);
  ~SuperclusValueMapProducer() override = default;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  std::vector<edm::EDGetTokenT<pat::PackedCandidateCollection>> setTokens(const std::vector<edm::InputTag>& tags);

  template <typename T>
  static void writeValueMap(edm::Event& iEvent,
                            const edm::Handle<edm::View<reco::SuperCluster>>& handle,
                            const std::vector<T>& values,
                            const std::string& label);

  std::vector<SuperclusTkIsolFromCands::PIDVeto> candVetos_;

  const std::vector<edm::InputTag> candTags_;
  const std::vector<edm::EDGetTokenT<pat::PackedCandidateCollection>> candTokens_;
  const edm::EDGetTokenT<edm::View<reco::SuperCluster>> scToken_;
  const edm::EDGetTokenT<edm::View<reco::Vertex>> pvToken_;
  const edm::EDGetTokenT<reco::BeamSpot> bsToken_;

  const SuperclusTkIsolFromCands::Configuration trkIsoCalcCfg_;

  const std::string superclusTkIsoLabel_ = "superclusTkIso";
};

SuperclusValueMapProducer::SuperclusValueMapProducer(const edm::ParameterSet& iConfig)
    : candTags_(iConfig.getParameter<std::vector<edm::InputTag>>("cands")),
      candTokens_(setTokens(candTags_)),
      scToken_(consumes<edm::View<reco::SuperCluster>>(iConfig.getParameter<edm::InputTag>("srcSc"))),
      pvToken_(consumes<edm::View<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("srcPv"))),
      bsToken_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("srcBs"))),
      trkIsoCalcCfg_(iConfig.getParameter<edm::ParameterSet>("trkIsoConfig")) {
  auto fillVetos = [](const auto& in, auto& out) {
    std::transform(in.begin(), in.end(), std::back_inserter(out), SuperclusTkIsolFromCands::pidVetoFromStr);
  };

  fillVetos(iConfig.getParameter<std::vector<std::string>>("candVetos"), candVetos_);

  if (candVetos_.size() != candTags_.size())
    throw cms::Exception("ConfigError") << "Error candVetos should be the same size as cands" << std::endl;

  produces<edm::ValueMap<float>>(superclusTkIsoLabel_);
}

void SuperclusValueMapProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<edm::View<reco::SuperCluster>> scHandle;
  iEvent.getByToken(scToken_, scHandle);

  edm::Handle<edm::View<reco::Vertex>> pvHandle;
  iEvent.getByToken(pvToken_, pvHandle);

  edm::Handle<reco::BeamSpot> bsHandle;
  iEvent.getByToken(bsToken_, bsHandle);

  math::XYZPoint pos;

  if (pvHandle.isValid() && !pvHandle->empty())
    pos = pvHandle->front().position();  // first try PV
  else
    pos = (*bsHandle).position();  // fall back to BS

  std::vector<edm::Handle<pat::PackedCandidateCollection>> candHandles(candTokens_.size());
  std::vector<std::unique_ptr<SuperclusTkIsolFromCands>> tkIsoCalc;

  for (unsigned idx = 0; idx < candTokens_.size(); idx++) {
    iEvent.getByToken(candTokens_.at(idx), candHandles.at(idx));
    tkIsoCalc.push_back(
        std::make_unique<SuperclusTkIsolFromCands>(trkIsoCalcCfg_, *(candHandles.at(idx)), candVetos_.at(idx)));
  }

  std::vector<float> vecTkIso;
  vecTkIso.reserve(scHandle->size());

  for (const auto& sc : *scHandle) {
    float tkIso = 0.;

    for (auto& calc : tkIsoCalc)
      tkIso += (*calc)(sc, pos).ptSum;

    vecTkIso.push_back(tkIso);
  }

  writeValueMap(iEvent, scHandle, vecTkIso, superclusTkIsoLabel_);
}

std::vector<edm::EDGetTokenT<pat::PackedCandidateCollection>> SuperclusValueMapProducer::setTokens(
    const std::vector<edm::InputTag>& tags) {
  std::vector<edm::EDGetTokenT<pat::PackedCandidateCollection>> out;

  out.reserve(tags.size());
  for (const auto& tag : tags)
    out.push_back(consumes<pat::PackedCandidateCollection>(tag));

  return out;
}

template <typename T>
void SuperclusValueMapProducer::writeValueMap(edm::Event& iEvent,
                                              const edm::Handle<edm::View<reco::SuperCluster>>& handle,
                                              const std::vector<T>& values,
                                              const std::string& label) {
  std::unique_ptr<edm::ValueMap<T>> valMap(new edm::ValueMap<T>());
  typename edm::ValueMap<T>::Filler filler(*valMap);
  filler.insert(handle, values.begin(), values.end());
  filler.fill();
  iEvent.put(std::move(valMap), label);
}

DEFINE_FWK_MODULE(SuperclusValueMapProducer);
