// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/Transition.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/HcalDigi/interface/HcalUMNioDigi.h"

class HcalUMNioTableProducer : public edm::stream::EDProducer<> {
private:
  edm::EDGetTokenT<HcalUMNioDigi> tokenUMNio_;
  edm::InputTag tagUMNio_;

public:
  explicit HcalUMNioTableProducer(const edm::ParameterSet& iConfig)
      : tagUMNio_(iConfig.getUntrackedParameter<edm::InputTag>("tagUMNio", edm::InputTag("hcalDigis"))) {
    tokenUMNio_ = consumes<HcalUMNioDigi>(tagUMNio_);

    produces<nanoaod::FlatTable>("uMNioTable");
  }

  ~HcalUMNioTableProducer() override{};

  /*
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
        edm::ParameterSetDescription desc;
        desc.add<edm::InputTag>("tagUMNio")->setComment("Input uMNio digi collection");
        descriptions.add("HcalUMNioTable", desc);
    }
    */

private:
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void produce(edm::Event&, edm::EventSetup const&) override;
};

void HcalUMNioTableProducer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {}

void HcalUMNioTableProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<HcalUMNioDigi> uMNioDigi;
  iEvent.getByToken(tokenUMNio_, uMNioDigi);
  uint8_t eventType = uMNioDigi->eventType();

  auto uMNioNanoTable = std::make_unique<nanoaod::FlatTable>(1, "uMNio", true);
  uMNioNanoTable->addColumnValue<uint8_t>("EventType", eventType, "EventType");
  for (int iWord = 0; iWord < uMNioDigi->numberUserWords(); ++iWord) {
    uint32_t thisWord = uMNioDigi->valueUserWord(iWord);
    uMNioNanoTable->addColumnValue<uint32_t>(
        "UserWord" + std::to_string(iWord), thisWord, "UserWord" + std::to_string(iWord));
  }
  iEvent.put(std::move(uMNioNanoTable), "uMNioTable");
}

#include "FWCore/Framework/interface/MakerMacros.h"
//define this as a plug-in
DEFINE_FWK_MODULE(HcalUMNioTableProducer);
