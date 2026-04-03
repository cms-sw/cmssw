/**
 * Run3ScoutingL1CaloProducer
 *
 * Copies L1 calorimeter objects (jets, EGamma, tau, EtSum) from gtStage2Digis
 * to produce collections with the standard caloStage2Digis module label
 * expected by downstream code.
 *
 * This allows scouting MiniAOD to be compatible with standard workflows
 * that expect L1 calo objects from caloStage2Digis.
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"

class Run3ScoutingL1CaloProducer : public edm::stream::EDProducer<> {
public:
  explicit Run3ScoutingL1CaloProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<BXVector<l1t::Jet>> jetToken_;
  const edm::EDGetTokenT<BXVector<l1t::EGamma>> egammaToken_;
  const edm::EDGetTokenT<BXVector<l1t::Tau>> tauToken_;
  const edm::EDGetTokenT<BXVector<l1t::EtSum>> etsumToken_;

  const edm::EDPutTokenT<BXVector<l1t::Jet>> jetPutToken_;
  const edm::EDPutTokenT<BXVector<l1t::EGamma>> egammaPutToken_;
  const edm::EDPutTokenT<BXVector<l1t::Tau>> tauPutToken_;
  const edm::EDPutTokenT<BXVector<l1t::EtSum>> etsumPutToken_;
};

Run3ScoutingL1CaloProducer::Run3ScoutingL1CaloProducer(const edm::ParameterSet& iConfig)
    : jetToken_(consumes<BXVector<l1t::Jet>>(iConfig.getParameter<edm::InputTag>("jetSource"))),
      egammaToken_(consumes<BXVector<l1t::EGamma>>(iConfig.getParameter<edm::InputTag>("egammaSource"))),
      tauToken_(consumes<BXVector<l1t::Tau>>(iConfig.getParameter<edm::InputTag>("tauSource"))),
      etsumToken_(consumes<BXVector<l1t::EtSum>>(iConfig.getParameter<edm::InputTag>("etsumSource"))),
      jetPutToken_(produces<BXVector<l1t::Jet>>("Jet")),
      egammaPutToken_(produces<BXVector<l1t::EGamma>>("EGamma")),
      tauPutToken_(produces<BXVector<l1t::Tau>>("Tau")),
      etsumPutToken_(produces<BXVector<l1t::EtSum>>("EtSum")) {}

void Run3ScoutingL1CaloProducer::produce(edm::Event& iEvent, const edm::EventSetup&) {
  // Copy jets
  {
    auto const& jetsIn = iEvent.get(jetToken_);
    BXVector<l1t::Jet> jetsOut;
    jetsOut.setBXRange(jetsIn.getFirstBX(), jetsIn.getLastBX());
    for (int bx = jetsIn.getFirstBX(); bx <= jetsIn.getLastBX(); ++bx) {
      for (auto it = jetsIn.begin(bx); it != jetsIn.end(bx); ++it) {
        jetsOut.push_back(bx, *it);
      }
    }
    iEvent.emplace(jetPutToken_, std::move(jetsOut));
  }

  // Copy EGamma
  {
    auto const& egammaIn = iEvent.get(egammaToken_);
    BXVector<l1t::EGamma> egammaOut;
    egammaOut.setBXRange(egammaIn.getFirstBX(), egammaIn.getLastBX());
    for (int bx = egammaIn.getFirstBX(); bx <= egammaIn.getLastBX(); ++bx) {
      for (auto it = egammaIn.begin(bx); it != egammaIn.end(bx); ++it) {
        egammaOut.push_back(bx, *it);
      }
    }
    iEvent.emplace(egammaPutToken_, std::move(egammaOut));
  }

  // Copy Tau
  {
    auto const& tauIn = iEvent.get(tauToken_);
    BXVector<l1t::Tau> tauOut;
    tauOut.setBXRange(tauIn.getFirstBX(), tauIn.getLastBX());
    for (int bx = tauIn.getFirstBX(); bx <= tauIn.getLastBX(); ++bx) {
      for (auto it = tauIn.begin(bx); it != tauIn.end(bx); ++it) {
        tauOut.push_back(bx, *it);
      }
    }
    iEvent.emplace(tauPutToken_, std::move(tauOut));
  }

  // Copy EtSum
  {
    auto const& etsumIn = iEvent.get(etsumToken_);
    BXVector<l1t::EtSum> etsumOut;
    etsumOut.setBXRange(etsumIn.getFirstBX(), etsumIn.getLastBX());
    for (int bx = etsumIn.getFirstBX(); bx <= etsumIn.getLastBX(); ++bx) {
      for (auto it = etsumIn.begin(bx); it != etsumIn.end(bx); ++it) {
        etsumOut.push_back(bx, *it);
      }
    }
    iEvent.emplace(etsumPutToken_, std::move(etsumOut));
  }
}

void Run3ScoutingL1CaloProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("jetSource", edm::InputTag("gtStage2Digis", "Jet"));
  desc.add<edm::InputTag>("egammaSource", edm::InputTag("gtStage2Digis", "EGamma"));
  desc.add<edm::InputTag>("tauSource", edm::InputTag("gtStage2Digis", "Tau"));
  desc.add<edm::InputTag>("etsumSource", edm::InputTag("gtStage2Digis", "EtSum"));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(Run3ScoutingL1CaloProducer);
