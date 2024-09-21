#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"

template <typename T>
class SimpleJetConstituentTableProducer : public edm::stream::EDProducer<> {
public:
  explicit SimpleJetConstituentTableProducer(const edm::ParameterSet &);
  ~SimpleJetConstituentTableProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::Event &, const edm::EventSetup &) override;

  const std::string name_;
  const std::string candIdxName_;
  const std::string candIdxDoc_;

  edm::EDGetTokenT<edm::View<T>> jet_token_;
  edm::EDGetTokenT<reco::CandidateView> cand_token_;

  const StringCutObjectSelector<T> jetCut_;

  edm::Handle<reco::CandidateView> cands_;
};

//
// constructors and destructor
//
template <typename T>
SimpleJetConstituentTableProducer<T>::SimpleJetConstituentTableProducer(const edm::ParameterSet &iConfig)
    : name_(iConfig.getParameter<std::string>("name")),
      candIdxName_(iConfig.getParameter<std::string>("candIdxName")),
      candIdxDoc_(iConfig.getParameter<std::string>("candIdxDoc")),
      jet_token_(consumes<edm::View<T>>(iConfig.getParameter<edm::InputTag>("jets"))),
      cand_token_(consumes<reco::CandidateView>(iConfig.getParameter<edm::InputTag>("candidates"))),
      jetCut_(iConfig.getParameter<std::string>("jetCut")) {
  produces<nanoaod::FlatTable>(name_);
  produces<std::vector<reco::CandidatePtr>>();
}

template <typename T>
SimpleJetConstituentTableProducer<T>::~SimpleJetConstituentTableProducer() {}

template <typename T>
void SimpleJetConstituentTableProducer<T>::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // elements in all these collections must have the same order!
  auto outCands = std::make_unique<std::vector<reco::CandidatePtr>>();

  auto jets = iEvent.getHandle(jet_token_);

  iEvent.getByToken(cand_token_, cands_);
  auto candPtrs = cands_->ptrs();

  //
  // First, select jets
  //
  std::vector<T> jetsPassCut;
  for (unsigned jetIdx = 0; jetIdx < jets->size(); ++jetIdx) {
    const auto &jet = jets->at(jetIdx);
    if (!jetCut_(jet))
      continue;
    jetsPassCut.push_back(jets->at(jetIdx));
  }

  //
  // Then loop over selected jets
  //
  std::vector<int> parentJetIdx;
  std::vector<int> candIdx;
  for (unsigned jetIdx = 0; jetIdx < jetsPassCut.size(); ++jetIdx) {
    const auto &jet = jetsPassCut.at(jetIdx);

    //
    // Loop over jet constituents
    //
    std::vector<reco::CandidatePtr> const &daughters = jet.daughterPtrVector();
    for (const auto &cand : daughters) {
      auto candInNewList = std::find(candPtrs.begin(), candPtrs.end(), cand);
      if (candInNewList == candPtrs.end()) {
        continue;
      }
      outCands->push_back(cand);
      parentJetIdx.push_back(jetIdx);
      candIdx.push_back(candInNewList - candPtrs.begin());
    }
  }  // end jet loop

  auto candTable = std::make_unique<nanoaod::FlatTable>(outCands->size(), name_, false);
  // We fill from here only stuff that cannot be created with the SimpleFlatTableProducer
  candTable->addColumn<int>(candIdxName_, candIdx, candIdxDoc_);

  std::string parentJetIdxName("jetIdx");
  std::string parentJetIdxDoc("Index of the parent jet");
  if constexpr (std::is_same<T, reco::GenJet>::value) {
    parentJetIdxName = "genJetIdx";
    parentJetIdxDoc = "Index of the parent gen jet";
  }
  candTable->addColumn<int>(parentJetIdxName, parentJetIdx, parentJetIdxDoc);

  iEvent.put(std::move(candTable), name_);
  iEvent.put(std::move(outCands));
}

template <typename T>
void SimpleJetConstituentTableProducer<T>::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("name", "FatJetPFCand");
  desc.add<std::string>("candIdxName", "PFCandIdx");
  desc.add<std::string>("candIdxDoc", "Index in PFCand table");
  desc.add<edm::InputTag>("jets", edm::InputTag("finalJetsAK8"));
  desc.add<edm::InputTag>("candidates", edm::InputTag("packedPFCandidates"));
  desc.add<std::string>("jetCut", "");
  descriptions.addWithDefaultLabel(desc);
}

typedef SimpleJetConstituentTableProducer<pat::Jet> SimplePatJetConstituentTableProducer;
typedef SimpleJetConstituentTableProducer<reco::GenJet> SimpleGenJetConstituentTableProducer;

DEFINE_FWK_MODULE(SimplePatJetConstituentTableProducer);
DEFINE_FWK_MODULE(SimpleGenJetConstituentTableProducer);