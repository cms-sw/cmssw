/**
 * ScoutingRhoProducer
 *
 * Simple producer that copies the rho value from scouting data
 * to produce it with standard module names expected by downstream code.
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

class ScoutingRhoProducer : public edm::stream::EDProducer<> {
public:
  explicit ScoutingRhoProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<double> rhoToken_;
};

ScoutingRhoProducer::ScoutingRhoProducer(const edm::ParameterSet& iConfig)
    : rhoToken_(consumes<double>(iConfig.getParameter<edm::InputTag>("src"))) {
  produces<double>();
}

void ScoutingRhoProducer::produce(edm::Event& iEvent, const edm::EventSetup&) {
  iEvent.put(std::make_unique<double>(iEvent.get(rhoToken_)));
}

void ScoutingRhoProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("hltScoutingPFPacker", "rho"));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(ScoutingRhoProducer);
