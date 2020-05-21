#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

class EmptySimHits : public edm::EDProducer {
public:
  explicit EmptySimHits(const edm::ParameterSet&);
  ~EmptySimHits() override{};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override{};
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endJob() override{};

  std::vector<std::string> pCaloHitInstanceLabels;
  std::vector<std::string> pSimHitInstanceLabels;
};

EmptySimHits::EmptySimHits(const edm::ParameterSet& iConfig) {
  pSimHitInstanceLabels = iConfig.getParameter<std::vector<std::string> >("pSimHitInstanceLabels");
  pCaloHitInstanceLabels = iConfig.getParameter<std::vector<std::string> >("pCaloHitInstanceLabels");

  for (const auto& pSimHitInstanceLabel : pSimHitInstanceLabels) {
    produces<edm::PSimHitContainer>(pSimHitInstanceLabel);
  }

  for (const auto& pCaloHitInstanceLabel : pCaloHitInstanceLabels) {
    produces<edm::PCaloHitContainer>(pCaloHitInstanceLabel);
  }
}

void EmptySimHits::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  for (const auto& pSimHitInstanceLabel : pSimHitInstanceLabels) {
    std::unique_ptr<edm::PSimHitContainer> pSimHitContainer(new edm::PSimHitContainer());
    iEvent.put(std::move(pSimHitContainer), pSimHitInstanceLabel);
  }

  for (const auto& pCaloHitInstanceLabel : pCaloHitInstanceLabels) {
    std::unique_ptr<edm::PCaloHitContainer> pCaloHitContainer(new edm::PCaloHitContainer());
    iEvent.put(std::move(pCaloHitContainer), pCaloHitInstanceLabel);
  }
}

void EmptySimHits::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(EmptySimHits);
