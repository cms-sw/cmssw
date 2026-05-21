#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

namespace pat {

  class EmbedMCinMuons : public edm::global::EDProducer<> {
  public:
    explicit EmbedMCinMuons(const edm::ParameterSet& iConfig)
        : muonToken_(consumes<pat::MuonCollection>(iConfig.getParameter<edm::InputTag>("muons"))),
          muonGenMatchToken_(consumes(iConfig.getParameter<edm::InputTag>("matchedGen"))) {
      produces<pat::MuonCollection>();
    }
    ~EmbedMCinMuons() override{};

    void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    const edm::EDGetTokenT<pat::MuonCollection> muonToken_;
    const edm::EDGetTokenT<edm::Association<reco::GenParticleCollection> > muonGenMatchToken_;
  };

}  // namespace pat

void pat::EmbedMCinMuons::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // extract input information
  const auto& matches = iEvent.get(muonGenMatchToken_);
  const auto& muons = iEvent.getHandle(muonToken_);

  // initialize output muon collection
  auto output = std::make_unique<pat::MuonCollection>(*muons);

  // add gen information to muons
  for (size_t i = 0; i < muons->size(); i++) {
    (*output)[i].setGenParticle(*matches.get(muons.id(), i));
  }

  iEvent.put(std::move(output));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void pat::EmbedMCinMuons::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("muons", edm::InputTag("unpackedMuons"))->setComment("muon input collection");
  desc.add<edm::InputTag>("matchedGen", edm::InputTag("muonMatch"))
      ->setComment("matches with gen muons input collection");
  descriptions.add("unpackedMuonsWithGenMatch", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(EmbedMCinMuons);
