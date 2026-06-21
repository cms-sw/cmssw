#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FTLDigi/interface/FTLDigiCollections.h"
#include "DataFormats/FTLRecHit/interface/FTLRecHitCollections.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/PluginDescription.h"

#include "RecoLocalFastTime/FTLCommonAlgos/interface/MTDUncalibratedRecHitAlgoBase.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

class MTDUncalibratedRecHitProducer : public edm::stream::EDProducer<> {
public:
  explicit MTDUncalibratedRecHitProducer(const edm::ParameterSet& ps);
  ~MTDUncalibratedRecHitProducer() override = default;
  void produce(edm::Event& evt, const edm::EventSetup& es) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::EDGetTokenT<BTLDigiCollection> ftlbDigis_;  // collection of BTL digis
  const edm::EDGetTokenT<ETLDigiCollection> ftleDigis_;  // collection of ETL digis

  const std::string ftlbInstance_;  // instance name of barrel hits
  const std::string ftleInstance_;  // instance name of endcap hits

  std::unique_ptr<BTLUncalibratedRecHitAlgoBase> barrel_;
  std::unique_ptr<ETLUncalibratedRecHitAlgoBase> endcap_;
};

MTDUncalibratedRecHitProducer::MTDUncalibratedRecHitProducer(const edm::ParameterSet& ps)
    : ftlbDigis_(consumes<BTLDigiCollection>(ps.getParameter<edm::InputTag>("barrelDigis"))),
      ftleDigis_(consumes<ETLDigiCollection>(ps.getParameter<edm::InputTag>("endcapDigis"))),
      ftlbInstance_(ps.getParameter<std::string>("BarrelHitsName")),
      ftleInstance_(ps.getParameter<std::string>("EndcapHitsName")) {
  produces<FTLUncalibratedRecHitCollection>(ftlbInstance_);
  produces<FTLUncalibratedRecHitCollection>(ftleInstance_);

  auto sumes = consumesCollector();

  const edm::ParameterSet& barrel = ps.getParameterSet("barrel");
  const std::string& barrelAlgo = barrel.getParameter<std::string>("algoName");
  barrel_ = std::unique_ptr<BTLUncalibratedRecHitAlgoBase>{
      BTLUncalibratedRecHitAlgoFactory::get()->create(barrelAlgo, barrel, sumes)};

  const edm::ParameterSet& endcap = ps.getParameterSet("endcap");
  const std::string& endcapAlgo = endcap.getParameter<std::string>("algoName");
  endcap_ = std::unique_ptr<ETLUncalibratedRecHitAlgoBase>{
      ETLUncalibratedRecHitAlgoFactory::get()->create(endcapAlgo, endcap, sumes)};
}

void MTDUncalibratedRecHitProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  // tranparently get things from event setup
  barrel_->getEventSetup(es);
  endcap_->getEventSetup(es);

  barrel_->getEvent(evt);
  endcap_->getEvent(evt);

  // prepare output
  auto barrelRechits = std::make_unique<FTLUncalibratedRecHitCollection>();
  auto endcapRechits = std::make_unique<FTLUncalibratedRecHitCollection>();

  edm::Handle<BTLDigiCollection> hBarrel;
  evt.getByToken(ftlbDigis_, hBarrel);
  if (hBarrel.isValid()) {
    barrelRechits->reserve(hBarrel->size() / 2);
    for (const auto& digi : *hBarrel) {
      barrelRechits->emplace_back(barrel_->makeRecHit(digi));
    }
  } else {
    edm::LogWarning("MTDReco") << "MTDUncalibratedRecHitProducer: Missing BTL Digi Collection";
  }

  edm::Handle<ETLDigiCollection> hEndcap;
  evt.getByToken(ftleDigis_, hEndcap);
  if (hEndcap.isValid()) {
    endcapRechits->reserve(hEndcap->size() / 2);
    for (const auto& digi : *hEndcap) {
      endcapRechits->emplace_back(endcap_->makeRecHit(digi));
    }
  } else {
    edm::LogWarning("MTDReco") << "MTDUncalibratedRecHitProducer: Missing ETL Digi Collection";
  }

  // put the collection of recunstructed hits in the event
  evt.put(std::move(barrelRechits), ftlbInstance_);
  evt.put(std::move(endcapRechits), ftleInstance_);
}

void MTDUncalibratedRecHitProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("barrelDigis", edm::InputTag("mix", "FTLBarrel"));
  desc.add<edm::InputTag>("endcapDigis", edm::InputTag("mix", "FTEndcap"));

  desc.add<std::string>("BarrelHitsName", "FTLBarrel");
  desc.add<std::string>("EndcapHitsName", "FTLEndcap");

  edm::ParameterSetDescription barrelDesc;
  barrelDesc.addNode(edm::PluginDescription<BTLUncalibratedRecHitAlgoFactory>("algoName", true));
  desc.add<edm::ParameterSetDescription>("barrel", barrelDesc);

  edm::ParameterSetDescription endcapDesc;
  endcapDesc.addNode(edm::PluginDescription<ETLUncalibratedRecHitAlgoFactory>("algoName", true));
  desc.add<edm::ParameterSetDescription>("endcap", endcapDesc);

  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MTDUncalibratedRecHitProducer);
