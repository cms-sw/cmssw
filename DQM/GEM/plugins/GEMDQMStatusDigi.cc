#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "EventFilter/GEMRawToDigi/interface/GEMVfatStatusDigiCollection.h"
#include "EventFilter/GEMRawToDigi/interface/GEMGEBdataCollection.h"
#include "EventFilter/GEMRawToDigi/interface/GEMAMCdataCollection.h"

#include <string>

//----------------------------------------------------------------------------------------------------

class GEMDQMStatusDigi : public DQMEDAnalyzer {
public:
  GEMDQMStatusDigi(const edm::ParameterSet &cfg);
  ~GEMDQMStatusDigi() override{};
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

protected:
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const &e, edm::EventSetup const &eSetup) override;
  void endRun(edm::Run const &run, edm::EventSetup const &eSetup) override{};

private:
  int nVfat_ = 24;
  int cBit_ = 9;
  int eBit_ = 13;
  edm::EDGetToken tagVFAT_;
  edm::EDGetToken tagGEB_;
  edm::EDGetToken tagAMC_;

  MonitorElement *h1_vfat_quality_;
  MonitorElement *h1_vfat_flag_;
  MonitorElement *h2_vfat_quality_;
  MonitorElement *h2_vfat_flag_;

  MonitorElement *h1_geb_inputStatus_;
  MonitorElement *h1_geb_vfatWordCnt_;
  MonitorElement *h1_geb_zeroSupWordsCnt_;
  MonitorElement *h1_geb_stuckData_;
  MonitorElement *h1_geb_inFIFOund_;

  MonitorElement *h1_amc_ttsState_;
  MonitorElement *h1_amc_davCnt_;
  MonitorElement *h1_amc_buffState_;
  MonitorElement *h1_amc_oosGlib_;
  MonitorElement *h1_amc_chTimeOut_;
};

using namespace std;
using namespace edm;

GEMDQMStatusDigi::GEMDQMStatusDigi(const edm::ParameterSet &cfg) {
  tagVFAT_ = consumes<GEMVfatStatusDigiCollection>(cfg.getParameter<edm::InputTag>("VFATInputLabel"));
  tagGEB_ = consumes<GEMGEBdataCollection>(cfg.getParameter<edm::InputTag>("GEBInputLabel"));
  tagAMC_ = consumes<GEMAMCdataCollection>(cfg.getParameter<edm::InputTag>("AMCInputLabel"));
}

void GEMDQMStatusDigi::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("VFATInputLabel", edm::InputTag("muonGEMDigis", "vfatStatus"));
  desc.add<edm::InputTag>("GEBInputLabel", edm::InputTag("muonGEMDigis", "GEBStatus"));
  desc.add<edm::InputTag>("AMCInputLabel", edm::InputTag("muonGEMDigis", "AMCStatus"));
  descriptions.add("GEMDQMStatusDigi", desc);
}

void GEMDQMStatusDigi::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const &, edm::EventSetup const &iSetup) {
  ibooker.cd();
  ibooker.setCurrentFolder("GEM/StatusDigi");

  h1_vfat_quality_ = ibooker.book1D("vfat quality", "quality", 6, 0, 6);
  h1_vfat_flag_ = ibooker.book1D("vfat flag", "flag", 5, 0, 5);

  h2_vfat_quality_ = ibooker.book2D("vfat quality per geb", "quality", 6, 0, 6, 36, 0, 36);
  h2_vfat_flag_ = ibooker.book2D("vfat flag per geb", "flag", 5, 0, 5, 36, 0, 36);

  h1_geb_inputStatus_ = ibooker.book1D("geb input status", "inputStatus", 10, 0, 10);
  h1_geb_vfatWordCnt_ = ibooker.book1D("geb no. vfats", "nvfats", 25, 0, 25);
  h1_geb_zeroSupWordsCnt_ = ibooker.book1D("geb zeroSupWordsCnt", "zeroSupWordsCnt", 10, 0, 10);
  h1_geb_stuckData_ = ibooker.book1D("geb stuckData", "stuckData", 10, 0, 10);
  h1_geb_inFIFOund_ = ibooker.book1D("geb inFIFOund", "inFIFOund", 10, 0, 10);

  h1_amc_ttsState_ = ibooker.book1D("amc ttsState", "ttsState", 10, 0, 10);
  h1_amc_davCnt_ = ibooker.book1D("amc davCnt", "davCnt", 10, 0, 10);
  h1_amc_buffState_ = ibooker.book1D("amc buffState", "buffState", 10, 0, 10);
  h1_amc_oosGlib_ = ibooker.book1D("amc oosGlib", "oosGlib", 10, 0, 10);
  h1_amc_chTimeOut_ = ibooker.book1D("amc chTimeOut", "chTimeOut", 10, 0, 10);
}

void GEMDQMStatusDigi::analyze(edm::Event const &event, edm::EventSetup const &eventSetup) {
  edm::Handle<GEMVfatStatusDigiCollection> gemVFAT;
  edm::Handle<GEMGEBdataCollection> gemGEB;
  edm::Handle<GEMAMCdataCollection> gemAMC;
  event.getByToken(tagVFAT_, gemVFAT);
  event.getByToken(tagGEB_, gemGEB);
  event.getByToken(tagAMC_, gemAMC);

  for (GEMVfatStatusDigiCollection::DigiRangeIterator vfatIt = gemVFAT->begin(); vfatIt != gemVFAT->end(); ++vfatIt) {
    GEMDetId gemid = (*vfatIt).first;
    float nIdx = gemid.chamber() + (gemid.layer() - 1) / 2.0;
    const GEMVfatStatusDigiCollection::Range &range = (*vfatIt).second;
    for (auto vfatStat = range.first; vfatStat != range.second; ++vfatStat) {
      h1_vfat_quality_->Fill(vfatStat->quality());
      h1_vfat_flag_->Fill(vfatStat->flag());
      h2_vfat_quality_->Fill(vfatStat->quality(), nIdx);
      h2_vfat_flag_->Fill(vfatStat->flag(), nIdx);
    }
  }

  for (GEMGEBdataCollection::DigiRangeIterator gebIt = gemGEB->begin(); gebIt != gemGEB->end(); ++gebIt) {
    const GEMGEBdataCollection::Range &range = (*gebIt).second;
    for (auto GEBStatus = range.first; GEBStatus != range.second; ++GEBStatus) {
      h1_geb_inputStatus_->Fill(GEBStatus->inputStatus());
      h1_geb_vfatWordCnt_->Fill(GEBStatus->vfatWordCnt() / 3);
      h1_geb_zeroSupWordsCnt_->Fill(GEBStatus->zeroSupWordsCnt());
      h1_geb_stuckData_->Fill(GEBStatus->stuckData());
      h1_geb_inFIFOund_->Fill(GEBStatus->inFIFOund());
    }
  }

  for (GEMAMCdataCollection::DigiRangeIterator amcIt = gemAMC->begin(); amcIt != gemAMC->end(); ++amcIt) {
    const GEMAMCdataCollection::Range &range = (*amcIt).second;
    for (auto amc = range.first; amc != range.second; ++amc) {
      h1_amc_ttsState_->Fill(amc->ttsState());
      h1_amc_davCnt_->Fill(amc->davCnt());
      h1_amc_buffState_->Fill(amc->buffState());
      h1_amc_oosGlib_->Fill(amc->oosGlib());
      h1_amc_chTimeOut_->Fill(amc->chTimeOut());
    }
  }
}

DEFINE_FWK_MODULE(GEMDQMStatusDigi);
