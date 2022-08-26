#include "DataFormats/L1TParticleFlow/interface/PFJet.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "L1Trigger/Phase2L1ParticleFlow/src/corrector.h"

#include <vector>

class L1TCorrectedPFJetProducer : public edm::global::EDProducer<> {
public:
  explicit L1TCorrectedPFJetProducer(const edm::ParameterSet&);
  ~L1TCorrectedPFJetProducer() override;

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  edm::EDGetTokenT<edm::View<l1t::PFJet>> jets_;
  l1tpf::corrector corrector_;
};

L1TCorrectedPFJetProducer::L1TCorrectedPFJetProducer(const edm::ParameterSet& iConfig)
    : jets_(consumes<edm::View<l1t::PFJet>>(iConfig.getParameter<edm::InputTag>("jets"))),
      corrector_(iConfig.getParameter<std::string>("correctorFile"),
                 iConfig.getParameter<std::string>("correctorDir")) {
  produces<std::vector<l1t::PFJet>>();
}

L1TCorrectedPFJetProducer::~L1TCorrectedPFJetProducer() {}

void L1TCorrectedPFJetProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  edm::Handle<edm::View<l1t::PFJet>> jets;
  iEvent.getByToken(jets_, jets);
  auto out = std::make_unique<std::vector<l1t::PFJet>>();

  for (const auto& srcjet : *jets) {
    // start out as copy
    out->emplace_back(srcjet.p4());
    auto& jet = out->back();
    // copy daughters
    for (const auto& dau : srcjet.constituents()) {
      jet.addConstituent(dau);
    }
    // apply corrections
    jet.calibratePt(corrector_.correctedPt(jet.pt(), jet.eta()));
  }

  iEvent.put(std::move(out));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TCorrectedPFJetProducer);
