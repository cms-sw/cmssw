#include <memory>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

class HiSignalGenJetProducer : public edm::global::EDProducer<> {
public:
  explicit HiSignalGenJetProducer(const edm::ParameterSet&);
  ~HiSignalGenJetProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  edm::EDGetTokenT<edm::View<reco::GenJet> > jetSrc_;
};

HiSignalGenJetProducer::HiSignalGenJetProducer(const edm::ParameterSet& iConfig)
    : jetSrc_(consumes<edm::View<reco::GenJet> >(iConfig.getParameter<edm::InputTag>("src"))) {
  std::string alias = (iConfig.getParameter<edm::InputTag>("src")).label();
  produces<reco::GenJetCollection>().setBranchAlias(alias);
}

void HiSignalGenJetProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  auto jets = std::make_unique<reco::GenJetCollection>();

  edm::Handle<edm::View<reco::GenJet> > genjets;
  iEvent.getByToken(jetSrc_, genjets);

  for (const reco::GenJet& jet1 : *genjets) {
    const reco::GenParticle* gencon = jet1.getGenConstituent(0);

    if (gencon == nullptr)
      throw cms::Exception("GenConstituent", "GenJet is missing its constituents");
    else if (gencon->collisionId() == 0) {
      jets->push_back(jet1);
    }
  }

  iEvent.put(std::move(jets));
}

void HiSignalGenJetProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Selects genJets from collision id = 0");
  desc.add<edm::InputTag>("src", edm::InputTag("akHiGenJets"));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(HiSignalGenJetProducer);
