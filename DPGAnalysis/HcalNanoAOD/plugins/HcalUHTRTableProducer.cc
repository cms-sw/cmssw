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

#include "EventFilter/HcalRawToDigi/interface/HcalUHTRData.h"
#include "EventFilter/HcalRawToDigi/interface/AMC13Header.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include <iostream>

class HcalUHTRTableProducer : public edm::stream::EDProducer<> {
private:
  edm::EDGetTokenT<FEDRawDataCollection> tokenRaw_;
  edm::InputTag tagRaw_;
  std::vector<int> fedUnpackList_;

public:
  explicit HcalUHTRTableProducer(const edm::ParameterSet& iConfig)
      : tagRaw_(iConfig.getParameter<edm::InputTag>("InputLabel")),
        fedUnpackList_(iConfig.getUntrackedParameter<std::vector<int>>("FEDs", std::vector<int>())) {
    tokenRaw_ = consumes<FEDRawDataCollection>(tagRaw_);
    produces<nanoaod::FlatTable>("uHTRTable");

    if (fedUnpackList_.empty()) {
      // VME range for back-compatibility
      for (int i = FEDNumbering::MINHCALFEDID; i <= FEDNumbering::MAXHCALFEDID; i++)
        fedUnpackList_.push_back(i);

      // uTCA range
      for (int i = FEDNumbering::MINHCALuTCAFEDID; i <= FEDNumbering::MAXHCALuTCAFEDID; i++)
        fedUnpackList_.push_back(i);
    }
  }

  ~HcalUHTRTableProducer() override{};

  /*
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
        edm::ParameterSetDescription desc;
        desc.add<edm::InputTag>("tagUHTR")->setComment("Input uMNio digi collection");
        descriptions.add("HcalUHTRTable", desc);
    }
    */

private:
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void produce(edm::Event&, edm::EventSetup const&) override;
};

void HcalUHTRTableProducer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {}

void HcalUHTRTableProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::vector<int> crate;
  std::vector<int> slot;
  std::vector<uint32_t> evn;
  std::vector<uint32_t> bcn;
  std::vector<uint32_t> orn;
  std::vector<int> eventType;

  edm::Handle<FEDRawDataCollection> raw;
  iEvent.getByToken(tokenRaw_, raw);
  for (std::vector<int>::const_iterator i = fedUnpackList_.begin(); i != fedUnpackList_.end(); i++) {
    const FEDRawData& fed = raw->FEDData(*i);
    hcal::AMC13Header const* hamc13 = (hcal::AMC13Header const*)fed.data();
    if (!hamc13) {
      continue;
    }
    int namc = hamc13->NAMC();
    for (int iamc = 0; iamc < namc; iamc++) {
      HcalUHTRData uhtr(hamc13->AMCPayload(iamc), hamc13->AMCSize(iamc));
      crate.push_back(uhtr.crateId());
      slot.push_back(uhtr.slot());
      evn.push_back(uhtr.l1ANumber());
      bcn.push_back(uhtr.bunchNumber());
      orn.push_back(uhtr.orbitNumber());
      eventType.push_back(uhtr.getEventType());
    }
  }

  auto uHTRNanoTable = std::make_unique<nanoaod::FlatTable>(crate.size(), "uHTR", false, false);
  uHTRNanoTable->addColumn<int>("crate", crate, "crate");
  uHTRNanoTable->addColumn<int>("slot", slot, "slot");
  uHTRNanoTable->addColumn<uint32_t>("evn", evn, "evn");
  uHTRNanoTable->addColumn<uint32_t>("bcn", bcn, "bcn");
  uHTRNanoTable->addColumn<uint32_t>("orn", orn, "orn");
  uHTRNanoTable->addColumn<int>("eventType", eventType, "eventType");

  iEvent.put(std::move(uHTRNanoTable), "uHTRTable");
}

#include "FWCore/Framework/interface/MakerMacros.h"
//define this as a plug-in
DEFINE_FWK_MODULE(HcalUHTRTableProducer);
