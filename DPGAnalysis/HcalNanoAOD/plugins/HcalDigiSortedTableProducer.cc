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

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"

#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"

#include "DPGAnalysis/HcalNanoAOD/interface/QIE11DigiSortedTable.h"
#include "DPGAnalysis/HcalNanoAOD/interface/QIE10DigiSortedTable.h"
#include "DPGAnalysis/HcalNanoAOD/interface/HODigiSortedTable.h"

class HcalDigiSortedTableProducer : public edm::stream::EDProducer<> {
private:
  std::map<HcalSubdetector, edm::Handle<std::vector<HcalDetId>>> dids_;

  //std::map<HcalSubdetector, std::vector<HcalElectronicsId> > eids_;
  static const std::vector<HcalSubdetector> subdets_;
  HcalElectronicsMap const* emap_;

  edm::InputTag tagHBDetIdList_;
  edm::InputTag tagHEDetIdList_;
  edm::InputTag tagHFDetIdList_;
  edm::InputTag tagHODetIdList_;

  edm::EDGetTokenT<std::vector<HcalDetId>> tokenHBDetIdList_;
  edm::EDGetTokenT<std::vector<HcalDetId>> tokenHEDetIdList_;
  edm::EDGetTokenT<std::vector<HcalDetId>> tokenHFDetIdList_;
  edm::EDGetTokenT<std::vector<HcalDetId>> tokenHODetIdList_;

  edm::InputTag tagQIE11_;
  edm::InputTag tagQIE10_;
  edm::InputTag tagHO_;

  edm::EDGetTokenT<QIE11DigiCollection> tokenQIE11_;
  edm::EDGetTokenT<QIE10DigiCollection> tokenQIE10_;
  edm::EDGetTokenT<HODigiCollection> tokenHO_;

  edm::ESGetToken<HcalDbService, HcalDbRecord> tokenHcalDbService_;
  edm::ESHandle<HcalDbService> dbService_;

  HBDigiSortedTable* hbDigiTable_;
  HEDigiSortedTable* heDigiTable_;
  HFDigiSortedTable* hfDigiTable_;
  HODigiSortedTable* hoDigiTable_;

  const unsigned int nTS_HB_;
  const unsigned int nTS_HE_;
  const unsigned int nTS_HF_;
  const unsigned int nTS_HO_;

public:
  explicit HcalDigiSortedTableProducer(const edm::ParameterSet& iConfig)
      : tokenHBDetIdList_(consumes<edm::InRun>(iConfig.getUntrackedParameter<edm::InputTag>(
            "HBDetIdList", edm::InputTag("hcalDetIdTable", "HBDetIdList")))),
        tokenHEDetIdList_(consumes<edm::InRun>(iConfig.getUntrackedParameter<edm::InputTag>(
            "HEDetIdList", edm::InputTag("hcalDetIdTable", "HEDetIdList")))),
        tokenHFDetIdList_(consumes<edm::InRun>(iConfig.getUntrackedParameter<edm::InputTag>(
            "HFDetIdList", edm::InputTag("hcalDetIdTable", "HFDetIdList")))),
        tokenHODetIdList_(consumes<edm::InRun>(iConfig.getUntrackedParameter<edm::InputTag>(
            "HODetIdList", edm::InputTag("hcalDetIdTable", "HODetIdList")))),
        tagQIE11_(iConfig.getUntrackedParameter<edm::InputTag>("tagQIE11", edm::InputTag("hcalDigis"))),
        tagQIE10_(iConfig.getUntrackedParameter<edm::InputTag>("tagQIE10", edm::InputTag("hcalDigis"))),
        tagHO_(iConfig.getUntrackedParameter<edm::InputTag>("tagHO", edm::InputTag("hcalDigis"))),
        tokenHcalDbService_(esConsumes<HcalDbService, HcalDbRecord, edm::Transition::BeginRun>()),
        nTS_HB_(iConfig.getUntrackedParameter<unsigned int>("nTS_HB", 8)),
        nTS_HE_(iConfig.getUntrackedParameter<unsigned int>("nTS_HE", 8)),
        nTS_HF_(iConfig.getUntrackedParameter<unsigned int>("nTS_HF", 3)),
        nTS_HO_(iConfig.getUntrackedParameter<unsigned int>("nTS_HO", 10)) {
    tokenQIE11_ = consumes<QIE11DigiCollection>(tagQIE11_);
    tokenHO_ = consumes<HODigiCollection>(tagHO_);
    tokenQIE10_ = consumes<QIE10DigiCollection>(tagQIE10_);

    produces<nanoaod::FlatTable>("HBDigiSortedTable");
    produces<nanoaod::FlatTable>("HEDigiSortedTable");
    produces<nanoaod::FlatTable>("HFDigiSortedTable");
    produces<nanoaod::FlatTable>("HODigiSortedTable");

    hbDigiTable_ = nullptr;
    heDigiTable_ = nullptr;
    hfDigiTable_ = nullptr;
    hoDigiTable_ = nullptr;
  }

  ~HcalDigiSortedTableProducer() override {
    delete hbDigiTable_;
    delete heDigiTable_;
    delete hfDigiTable_;
    delete hoDigiTable_;
  };

  /*
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
        edm::ParameterSetDescription desc;
        desc.add<edm::InputTag>("tagQIE11")->setComment("Input QIE 11 digi collection");
        // desc.add<std::string>("name")->setComment("");
        descriptions.add("HcalDigiTable", desc);
    }
    */

private:
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void produce(edm::Event&, edm::EventSetup const&) override;
};

const std::vector<HcalSubdetector> HcalDigiSortedTableProducer::subdets_ = {
    HcalBarrel, HcalEndcap, HcalForward, HcalOuter};

void HcalDigiSortedTableProducer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  // List DetIds of interest from emap
  dbService_ = iSetup.getHandle(tokenHcalDbService_);
  emap_ = dbService_->getHcalMapping();

  iRun.getByToken(tokenHBDetIdList_, dids_[HcalBarrel]);
  iRun.getByToken(tokenHEDetIdList_, dids_[HcalEndcap]);
  iRun.getByToken(tokenHFDetIdList_, dids_[HcalForward]);
  iRun.getByToken(tokenHODetIdList_, dids_[HcalOuter]);

  // Create persistent, sorted digi storage
  hbDigiTable_ = new HBDigiSortedTable(*(dids_[HcalBarrel]), nTS_HB_);
  heDigiTable_ = new HEDigiSortedTable(*(dids_[HcalEndcap]), nTS_HE_);
  hfDigiTable_ = new HFDigiSortedTable(*(dids_[HcalForward]), nTS_HF_);
  hoDigiTable_ = new HODigiSortedTable(*(dids_[HcalOuter]), nTS_HO_);
}

void HcalDigiSortedTableProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // * Load digis */
  edm::Handle<QIE11DigiCollection> qie11Digis;
  iEvent.getByToken(tokenQIE11_, qie11Digis);

  edm::Handle<QIE10DigiCollection> qie10Digis;
  iEvent.getByToken(tokenQIE10_, qie10Digis);

  edm::Handle<HODigiCollection> hoDigis;
  iEvent.getByToken(tokenHO_, hoDigis);

  // * Process digis */
  // HB
  hbDigiTable_->reset();
  for (QIE11DigiCollection::const_iterator itDigi = qie11Digis->begin(); itDigi != qie11Digis->end(); ++itDigi) {
    const QIE11DataFrame digi = static_cast<const QIE11DataFrame>(*itDigi);
    HcalDetId const& did = digi.detid();
    if (did.subdet() != HcalBarrel)
      continue;

    hbDigiTable_->add(&digi, dbService_);
  }  // End loop over qie11 HB digis

  // HE
  heDigiTable_->reset();
  for (QIE11DigiCollection::const_iterator itDigi = qie11Digis->begin(); itDigi != qie11Digis->end(); ++itDigi) {
    const QIE11DataFrame digi = static_cast<const QIE11DataFrame>(*itDigi);
    HcalDetId const& did = digi.detid();
    if (did.subdet() != HcalEndcap)
      continue;

    heDigiTable_->add(&digi, dbService_);
  }  // End loop over qie11 HE digis

  // HF
  hfDigiTable_->reset();
  for (QIE10DigiCollection::const_iterator itDigi = qie10Digis->begin(); itDigi != qie10Digis->end(); ++itDigi) {
    const QIE10DataFrame digi = static_cast<const QIE10DataFrame>(*itDigi);
    HcalDetId const& did = digi.detid();
    if (did.subdet() != HcalForward)
      continue;

    hfDigiTable_->add(&digi, dbService_);
  }  // End loop over qie10 HF digis

  // HO
  hoDigiTable_->reset();
  for (HODigiCollection::const_iterator itDigi = hoDigis->begin(); itDigi != hoDigis->end(); ++itDigi) {
    const HODataFrame digi = static_cast<const HODataFrame>(*itDigi);
    HcalDetId const& did = digi.id();
    if (did.subdet() != HcalOuter)
      continue;

    hoDigiTable_->add(&digi, dbService_);
  }  // End loop over HO digis

  // * Save to NanoAOD tables */

  // HB
  auto hbNanoTable = std::make_unique<nanoaod::FlatTable>(dids_[HcalBarrel]->size(), "DigiHB", false, false);
  hbNanoTable->addColumn<int>("rawId", hbDigiTable_->rawIds_, "rawId");
  hbNanoTable->addColumn<int>("ieta", hbDigiTable_->ietas_, "ieta");
  hbNanoTable->addColumn<int>("iphi", hbDigiTable_->iphis_, "iphi");
  hbNanoTable->addColumn<int>("depth", hbDigiTable_->depths_, "depth");
  hbNanoTable->addColumn<int>("subdet", hbDigiTable_->subdets_, "subdet");
  hbNanoTable->addColumn<bool>("linkError", hbDigiTable_->linkErrors_, "linkError");
  hbNanoTable->addColumn<bool>("capidError", hbDigiTable_->capidErrors_, "capidError");
  hbNanoTable->addColumn<int>("flags", hbDigiTable_->flags_, "flags");
  hbNanoTable->addColumn<int>("soi", hbDigiTable_->sois_, "soi");
  hbNanoTable->addColumn<bool>("valid", hbDigiTable_->valids_, "valid");
  hbNanoTable->addColumn<uint8_t>("sipmTypes", hbDigiTable_->sipmTypes_, "sipmTypes");

  for (unsigned int iTS = 0; iTS < 8; ++iTS) {
    hbNanoTable->addColumn<int>(
        std::string("adc") + std::to_string(iTS), hbDigiTable_->adcs_[iTS], std::string("adc") + std::to_string(iTS));
    hbNanoTable->addColumn<int>(
        std::string("tdc") + std::to_string(iTS), hbDigiTable_->tdcs_[iTS], std::string("tdc") + std::to_string(iTS));
    hbNanoTable->addColumn<int>(std::string("capid") + std::to_string(iTS),
                                hbDigiTable_->capids_[iTS],
                                std::string("capid") + std::to_string(iTS));
    hbNanoTable->addColumn<float>(
        std::string("fc") + std::to_string(iTS), hbDigiTable_->fcs_[iTS], std::string("fc") + std::to_string(iTS));
    hbNanoTable->addColumn<float>(std::string("pedestalfc") + std::to_string(iTS),
                                  hbDigiTable_->pedestalfcs_[iTS],
                                  std::string("pedestalfc") + std::to_string(iTS));
  }
  iEvent.put(std::move(hbNanoTable), "HBDigiSortedTable");

  // HE
  auto heNanoTable = std::make_unique<nanoaod::FlatTable>(dids_[HcalEndcap]->size(), "DigiHE", false, false);
  heNanoTable->addColumn<int>("rawId", heDigiTable_->rawIds_, "rawId");
  heNanoTable->addColumn<int>("ieta", heDigiTable_->ietas_, "ieta");
  heNanoTable->addColumn<int>("iphi", heDigiTable_->iphis_, "iphi");
  heNanoTable->addColumn<int>("depth", heDigiTable_->depths_, "depth");
  heNanoTable->addColumn<int>("subdet", heDigiTable_->subdets_, "subdet");
  heNanoTable->addColumn<bool>("linkError", heDigiTable_->linkErrors_, "linkError");
  heNanoTable->addColumn<bool>("capidError", heDigiTable_->capidErrors_, "capidError");
  heNanoTable->addColumn<int>("flags", heDigiTable_->flags_, "flags");
  heNanoTable->addColumn<int>("soi", heDigiTable_->sois_, "soi");
  heNanoTable->addColumn<bool>("valid", heDigiTable_->valids_, "valid");
  heNanoTable->addColumn<uint8_t>("sipmTypes", heDigiTable_->sipmTypes_, "sipmTypes");

  for (unsigned int iTS = 0; iTS < 8; ++iTS) {
    heNanoTable->addColumn<int>(
        std::string("adc") + std::to_string(iTS), heDigiTable_->adcs_[iTS], std::string("adc") + std::to_string(iTS));
    heNanoTable->addColumn<int>(
        std::string("tdc") + std::to_string(iTS), heDigiTable_->tdcs_[iTS], std::string("tdc") + std::to_string(iTS));
    heNanoTable->addColumn<int>(std::string("capid") + std::to_string(iTS),
                                heDigiTable_->capids_[iTS],
                                std::string("capid") + std::to_string(iTS));
    heNanoTable->addColumn<float>(
        std::string("fc") + std::to_string(iTS), heDigiTable_->fcs_[iTS], std::string("fc") + std::to_string(iTS));
    heNanoTable->addColumn<float>(std::string("pedestalfc") + std::to_string(iTS),
                                  heDigiTable_->pedestalfcs_[iTS],
                                  std::string("pedestalfc") + std::to_string(iTS));
  }
  iEvent.put(std::move(heNanoTable), "HEDigiSortedTable");

  // HF
  auto hfNanoTable = std::make_unique<nanoaod::FlatTable>(dids_[HcalForward]->size(), "DigiHF", false, false);
  hfNanoTable->addColumn<int>("rawId", hfDigiTable_->rawIds_, "rawId");
  hfNanoTable->addColumn<int>("ieta", hfDigiTable_->ietas_, "ieta");
  hfNanoTable->addColumn<int>("iphi", hfDigiTable_->iphis_, "iphi");
  hfNanoTable->addColumn<int>("depth", hfDigiTable_->depths_, "depth");
  hfNanoTable->addColumn<int>("subdet", hfDigiTable_->subdets_, "subdet");
  hfNanoTable->addColumn<bool>("linkError", hfDigiTable_->linkErrors_, "linkError");
  hfNanoTable->addColumn<int>("flags", hfDigiTable_->flags_, "flags");
  hfNanoTable->addColumn<int>("soi", hfDigiTable_->sois_, "soi");
  hfNanoTable->addColumn<bool>("valid", hfDigiTable_->valids_, "valid");
  //hfNanoTable->addColumn<uint8_t>("sipmTypes", hfDigiTable_->sipmTypes_, "sipmTypes");

  for (unsigned int iTS = 0; iTS < 3; ++iTS) {
    hfNanoTable->addColumn<int>(
        std::string("adc") + std::to_string(iTS), hfDigiTable_->adcs_[iTS], std::string("adc") + std::to_string(iTS));
    hfNanoTable->addColumn<int>(
        std::string("tdc") + std::to_string(iTS), hfDigiTable_->tdcs_[iTS], std::string("tdc") + std::to_string(iTS));
    //hfNanoTable->addColumn<int>(std::string("tetdc") + std::to_string(iTS),
    //                                hfDigiTable_->tetdcs_[iTS],
    //                                std::string("tetdc") + std::to_string(iTS));
    hfNanoTable->addColumn<int>(std::string("capid") + std::to_string(iTS),
                                hfDigiTable_->capids_[iTS],
                                std::string("capid") + std::to_string(iTS));
    hfNanoTable->addColumn<float>(
        std::string("fc") + std::to_string(iTS), hfDigiTable_->fcs_[iTS], std::string("fc") + std::to_string(iTS));
    hfNanoTable->addColumn<float>(std::string("pedestalfc") + std::to_string(iTS),
                                  hfDigiTable_->pedestalfcs_[iTS],
                                  std::string("pedestalfc") + std::to_string(iTS));
    hfNanoTable->addColumn<float>(
        std::string("ok") + std::to_string(iTS), hfDigiTable_->oks_[iTS], std::string("ok") + std::to_string(iTS));
  }
  iEvent.put(std::move(hfNanoTable), "HFDigiSortedTable");

  // HO
  auto hoNanoTable = std::make_unique<nanoaod::FlatTable>(dids_[HcalOuter]->size(), "DigiHO", false, false);
  hoNanoTable->addColumn<int>("rawId", hoDigiTable_->rawIds_, "rawId");
  hoNanoTable->addColumn<int>("ieta", hoDigiTable_->ietas_, "ieta");
  hoNanoTable->addColumn<int>("iphi", hoDigiTable_->iphis_, "iphi");
  hoNanoTable->addColumn<int>("depth", hoDigiTable_->depths_, "depth");
  hoNanoTable->addColumn<int>("subdet", hoDigiTable_->subdets_, "subdet");
  hoNanoTable->addColumn<int>("fiberIdleOffset", hoDigiTable_->fiberIdleOffsets_, "fiberIdleOffset");
  hoNanoTable->addColumn<int>("soi", hoDigiTable_->sois_, "soi");
  hoNanoTable->addColumn<bool>("valid", hoDigiTable_->valids_, "valid");

  for (unsigned int iTS = 0; iTS < 10; ++iTS) {
    hoNanoTable->addColumn<int>(
        std::string("adc") + std::to_string(iTS), hoDigiTable_->adcs_[iTS], std::string("adc") + std::to_string(iTS));
    hoNanoTable->addColumn<int>(std::string("capid") + std::to_string(iTS),
                                hoDigiTable_->capids_[iTS],
                                std::string("capid") + std::to_string(iTS));
    hoNanoTable->addColumn<float>(
        std::string("fc") + std::to_string(iTS), hoDigiTable_->fcs_[iTS], std::string("fc") + std::to_string(iTS));
    hoNanoTable->addColumn<float>(std::string("pedestalfc") + std::to_string(iTS),
                                  hoDigiTable_->pedestalfcs_[iTS],
                                  std::string("pedestalfc") + std::to_string(iTS));
    hoNanoTable->addColumn<int>(std::string("fiber") + std::to_string(iTS),
                                hoDigiTable_->fibers_[iTS],
                                std::string("fiber") + std::to_string(iTS));
    hoNanoTable->addColumn<int>(std::string("fiberChan") + std::to_string(iTS),
                                hoDigiTable_->fiberChans_[iTS],
                                std::string("fiberChan") + std::to_string(iTS));
    hoNanoTable->addColumn<int>(
        std::string("dv") + std::to_string(iTS), hoDigiTable_->dvs_[iTS], std::string("dv") + std::to_string(iTS));
    hoNanoTable->addColumn<int>(
        std::string("er") + std::to_string(iTS), hoDigiTable_->ers_[iTS], std::string("er") + std::to_string(iTS));
  }
  iEvent.put(std::move(hoNanoTable), "HODigiSortedTable");
}

#include "FWCore/Framework/interface/MakerMacros.h"
//define this as a plug-in
DEFINE_FWK_MODULE(HcalDigiSortedTableProducer);
