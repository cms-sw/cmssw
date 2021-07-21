// system include files
#include <memory>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonShower.h"
#include "DataFormats/L1Trigger/interface/MuonShower.h"

using namespace l1t;

class L1TMuonShowerProducer : public edm::stream::EDProducer<> {
public:
  explicit L1TMuonShowerProducer(const edm::ParameterSet&);
  ~L1TMuonShowerProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  edm::InputTag showerInputTag_;
  edm::EDGetTokenT<l1t::RegionalMuonShowerBxCollection> showerInputToken_;
  int bxMin_;
  int bxMax_;
};

L1TMuonShowerProducer::L1TMuonShowerProducer(const edm::ParameterSet& iConfig)
    : showerInputTag_(iConfig.getParameter<edm::InputTag>("showerInput")),
      showerInputToken_(consumes<l1t::RegionalMuonShowerBxCollection>(showerInputTag_)),
      bxMin_(iConfig.getParameter<int>("bxMin")),
      bxMax_(iConfig.getParameter<int>("bxMax")) {
  produces<MuonShowerBxCollection>();
}

L1TMuonShowerProducer::~L1TMuonShowerProducer() {}

// ------------ method called to produce the data  ------------
void L1TMuonShowerProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  std::unique_ptr<MuonShowerBxCollection> outShowers(new MuonShowerBxCollection());

  Handle<l1t::RegionalMuonShowerBxCollection> emtfShowers;
  iEvent.getByToken(showerInputToken_, emtfShowers);
  outShowers->setBXRange(bxMin_, bxMax_);

  /*
    Check each sector for a valid EMTF shower. A valid EMTF shower
    can either be in-time or out-of-time. The minimal implementation
    only considers the "at least 1-nominal shower" case.
   */
  bool isOneNominalInTime = false;
  bool isOneNominalOutOfTime = false;
  bool isTwoLooseInTime = false;
  bool isTwoLooseOutOfTime = false;
  for (size_t i = 0; i < emtfShowers->size(0); ++i) {
    auto shower = emtfShowers->at(0, i);
    if (shower.isValid()) {
      // nominal
      if (shower.isOneNominalInTime())
        isOneNominalInTime = true;
      if (shower.isOneNominalOutOfTime())
        isOneNominalOutOfTime = true;
      // two loose
      if (shower.isTwoLooseInTime())
        isTwoLooseInTime = true;
      if (shower.isTwoLooseOutOfTime())
        isTwoLooseOutOfTime = true;
    }
  }

  // Check for at least one nominal shower
  const bool acceptCondition(isOneNominalInTime or isOneNominalOutOfTime or isTwoLooseInTime or isTwoLooseOutOfTime);
  if (acceptCondition) {
    MuonShower outShower(isOneNominalInTime, isOneNominalOutOfTime, isTwoLooseInTime, isTwoLooseOutOfTime);
    outShowers->push_back(0, outShower);
  }
  iEvent.put(std::move(outShowers));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void L1TMuonShowerProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("showerInput", edm::InputTag("simEmtfShowers", "EMTF"));
  desc.add<int32_t>("bxMin", 0);
  desc.add<int32_t>("bxMax", 0);
  desc.add<uint32_t>("minNominalShowers", 1);
  desc.add<uint32_t>("minTwoLooseShowers", 0);
  descriptions.add("simGmtShowerDigisDef", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TMuonShowerProducer);
