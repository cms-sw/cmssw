
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"

//
// class declaration
//

class GenJetGenPartMerger : public edm::stream::EDProducer<> {
public:
  explicit GenJetGenPartMerger(const edm::ParameterSet&);
  ~GenJetGenPartMerger() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;

  const edm::EDGetTokenT<reco::GenJetCollection> jetToken_;
  const edm::EDGetTokenT<reco::GenParticleCollection> partToken_;
  const StringCutObjectSelector<reco::Candidate> cut_;
  const edm::EDGetTokenT<edm::ValueMap<bool>> tauAncToken_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
GenJetGenPartMerger::GenJetGenPartMerger(const edm::ParameterSet& iConfig)
    : jetToken_(consumes<reco::GenJetCollection>(iConfig.getParameter<edm::InputTag>("srcJet"))),
      partToken_(consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("srcPart"))),
      cut_(iConfig.getParameter<std::string>("cut")),
      tauAncToken_(consumes<edm::ValueMap<bool>>(iConfig.getParameter<edm::InputTag>("hasTauAnc"))) {
  produces<reco::GenJetCollection>("merged");
  produces<edm::ValueMap<bool>>("hasTauAnc");
}

GenJetGenPartMerger::~GenJetGenPartMerger() {}

void GenJetGenPartMerger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("srcJet")->setComment("reco::GenJetCollection input collection");
  desc.add<edm::InputTag>("srcPart")->setComment("reco::GenParticleCollection input collection");
  desc.add<std::string>("cut")->setComment("a selection to apply to GenJet");
  desc.add<edm::InputTag>("hasTauAnc")->setComment("value map defining GenJet tau origin");
  descriptions.addWithDefaultLabel(desc);
}
//
// member functions
//

// ------------ method called to produce the data  ------------
void GenJetGenPartMerger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  auto merged = std::make_unique<reco::GenJetCollection>();

  std::vector<bool> hasTauAncValues;

  auto jetHandle = iEvent.getHandle(jetToken_);
  const auto& partProd = iEvent.get(partToken_);
  const auto& tauAncProd = iEvent.get(tauAncToken_);

  for (unsigned int ijet = 0; ijet < jetHandle->size(); ++ijet) {
    auto jet = jetHandle->at(ijet);
    if (cut_(jet)) {
      merged->push_back(reco::GenJet(jet));
      reco::GenJetRef jetRef(jetHandle, ijet);
      hasTauAncValues.push_back(tauAncProd[jetRef]);
    }
  }

  for (const auto& part : partProd) {
    reco::GenJet jet;
    jet.setP4(part.p4());
    jet.setPdgId(part.pdgId());
    jet.setCharge(part.charge());
    merged->push_back(jet);
    hasTauAncValues.push_back(false);
  }

  auto newmerged = iEvent.put(std::move(merged), "merged");

  auto out = std::make_unique<edm::ValueMap<bool>>();
  edm::ValueMap<bool>::Filler filler(*out);
  filler.insert(newmerged, hasTauAncValues.begin(), hasTauAncValues.end());
  filler.fill();

  iEvent.put(std::move(out), "hasTauAnc");
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void GenJetGenPartMerger::beginStream(edm::StreamID) {}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void GenJetGenPartMerger::endStream() {}

//define this as a plug-in
DEFINE_FWK_MODULE(GenJetGenPartMerger);
