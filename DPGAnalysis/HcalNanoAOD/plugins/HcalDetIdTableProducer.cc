#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/Transition.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"

#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"

class HcalDetIdTableProducer : public edm::global::EDProducer<edm::BeginRunProducer> {
private:
  edm::ESGetToken<HcalDbService, HcalDbRecord> tokenHcalDbService_;
  edm::EDPutTokenT<std::vector<HcalDetId>> hbDetIdListToken_;
  edm::EDPutTokenT<std::vector<HcalDetId>> heDetIdListToken_;
  edm::EDPutTokenT<std::vector<HcalDetId>> hfDetIdListToken_;
  edm::EDPutTokenT<std::vector<HcalDetId>> hoDetIdListToken_;

public:
  explicit HcalDetIdTableProducer(const edm::ParameterSet& iConfig)
      : tokenHcalDbService_(esConsumes<HcalDbService, HcalDbRecord, edm::Transition::BeginRun>()) {
    hbDetIdListToken_ = produces<std::vector<HcalDetId>, edm::Transition::BeginRun>("HBDetIdList");
    heDetIdListToken_ = produces<std::vector<HcalDetId>, edm::Transition::BeginRun>("HEDetIdList");
    hfDetIdListToken_ = produces<std::vector<HcalDetId>, edm::Transition::BeginRun>("HFDetIdList");
    hoDetIdListToken_ = produces<std::vector<HcalDetId>, edm::Transition::BeginRun>("HODetIdList");

    produces<nanoaod::FlatTable, edm::Transition::BeginRun>("HBDetIdList");
    produces<nanoaod::FlatTable, edm::Transition::BeginRun>("HEDetIdList");
    produces<nanoaod::FlatTable, edm::Transition::BeginRun>("HFDetIdList");
    produces<nanoaod::FlatTable, edm::Transition::BeginRun>("HODetIdList");
  };

  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
  void globalBeginRunProduce(edm::Run& iRun, edm::EventSetup const& iSetup) const override;
};

void HcalDetIdTableProducer::produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const {}

void HcalDetIdTableProducer::globalBeginRunProduce(edm::Run& iRun, edm::EventSetup const& iSetup) const {
  // Setup products
  const std::vector<HcalSubdetector> subdets = {HcalBarrel, HcalEndcap, HcalForward, HcalOuter};
  std::map<HcalSubdetector, std::unique_ptr<std::vector<HcalDetId>>> didLists;
  didLists[HcalBarrel] = std::make_unique<std::vector<HcalDetId>>();
  didLists[HcalEndcap] = std::make_unique<std::vector<HcalDetId>>();
  didLists[HcalForward] = std::make_unique<std::vector<HcalDetId>>();
  didLists[HcalOuter] = std::make_unique<std::vector<HcalDetId>>();

  // Load channels from emap
  edm::ESHandle<HcalDbService> dbService = iSetup.getHandle(tokenHcalDbService_);
  HcalElectronicsMap const* emap = dbService->getHcalMapping();

  std::vector<HcalGenericDetId> alldids = emap->allPrecisionId();
  for (auto it_did = alldids.begin(); it_did != alldids.end(); ++it_did) {
    if (!it_did->isHcalDetId()) {
      continue;
    }
    HcalDetId did = HcalDetId(it_did->rawId());
    if (!(did.subdet() == HcalBarrel || did.subdet() == HcalEndcap || did.subdet() == HcalForward ||
          did.subdet() == HcalOuter)) {
      continue;
    }

    // TODO: Add filtering, for example on FED whitelist

    didLists[did.subdet()]->push_back(did);
  }

  // Sort HcalDetIds
  for (auto& it_subdet : subdets) {
    std::sort(didLists[it_subdet]->begin(), didLists[it_subdet]->end());
  }

  // Make NanoAOD tables
  std::map<HcalSubdetector, std::string> subdetNames = {
      {HcalBarrel, "HB"}, {HcalEndcap, "HE"}, {HcalForward, "HF"}, {HcalOuter, "HO"}};

  for (auto& it_subdet : subdets) {
    auto didTable =
        std::make_unique<nanoaod::FlatTable>(didLists[it_subdet]->size(), subdetNames[it_subdet], false, false);

    std::vector<int> vdids;
    for (auto& it_did : *(didLists[it_subdet])) {
      vdids.push_back(it_did.rawId());
    }
    didTable->addColumn<int>("did", vdids, "HcalDetId");

    iRun.put(std::move(didTable), subdetNames[it_subdet] + "DetIdList");
    iRun.put(std::move(didLists[it_subdet]), subdetNames[it_subdet] + "DetIdList");
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
//define this as a plug-in
DEFINE_FWK_MODULE(HcalDetIdTableProducer);
