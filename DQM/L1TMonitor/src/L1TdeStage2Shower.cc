#include <string>

#include "DQM/L1TMonitor/interface/L1TdeStage2Shower.h"

L1TdeStage2Shower::L1TdeStage2Shower(const edm::ParameterSet& ps)
    : data_EMTFShower_token_(
          consumes<l1t::RegionalMuonShowerBxCollection>(ps.getParameter<edm::InputTag>("dataSource"))),
      emul_EMTFShower_token_(
          consumes<l1t::RegionalMuonShowerBxCollection>(ps.getParameter<edm::InputTag>("emulSource"))),
      monitorDir_(ps.getUntrackedParameter<std::string>("monitorDir")) {}

L1TdeStage2Shower::~L1TdeStage2Shower() {}

void L1TdeStage2Shower::bookHistograms(DQMStore::IBooker& iBooker, const edm::Run&, const edm::EventSetup&) {
  iBooker.setCurrentFolder(monitorDir_);

  // 2D summary plots
  emtfShowerDataSummary_denom_ =
      iBooker.book2D("emtf_shower_data_summary_denom", "Data EMTF Shower All", 6, 1, 7, 2, 0, 2);
  emtfShowerDataSummary_num_ =
      iBooker.book2D("emtf_shower_data_summary_num", "Data EMTF Shower Emul Matched", 6, 1, 7, 2, 0, 2);
  emtfShowerEmulSummary_denom_ =
      iBooker.book2D("emtf_shower_emul_summary_denom", "Emul EMTF Shower All", 6, 1, 7, 2, 0, 2);
  emtfShowerEmulSummary_num_ =
      iBooker.book2D("emtf_shower_emul_summary_num", "Emul EMTF Shower Not Matched to Data", 6, 1, 7, 2, 0, 2);

  // x labels
  emtfShowerDataSummary_denom_->setAxisTitle("Sector", 1);
  emtfShowerDataSummary_num_->setAxisTitle("Sector", 1);
  emtfShowerEmulSummary_denom_->setAxisTitle("Sector", 1);
  emtfShowerEmulSummary_num_->setAxisTitle("Sector", 1);

  // plotting option
  emtfShowerDataSummary_denom_->setOption("colz");
  emtfShowerDataSummary_num_->setOption("colz");
  emtfShowerEmulSummary_denom_->setOption("colz");
  emtfShowerEmulSummary_num_->setOption("colz");

  // y labels
  emtfShowerDataSummary_denom_->setBinLabel(1, "ME-", 2);
  emtfShowerDataSummary_num_->setBinLabel(1, "ME-", 2);
  emtfShowerEmulSummary_denom_->setBinLabel(1, "ME-", 2);
  emtfShowerEmulSummary_num_->setBinLabel(1, "ME-", 2);

  emtfShowerDataSummary_denom_->setBinLabel(2, "ME+", 2);
  emtfShowerDataSummary_num_->setBinLabel(2, "ME+", 2);
  emtfShowerEmulSummary_denom_->setBinLabel(2, "ME+", 2);
  emtfShowerEmulSummary_num_->setBinLabel(2, "ME+", 2);
}

void L1TdeStage2Shower::analyze(const edm::Event& e, const edm::EventSetup& c) {
  edm::Handle<l1t::RegionalMuonShowerBxCollection> dataShowers;
  edm::Handle<l1t::RegionalMuonShowerBxCollection> emulShowers;

  e.getByToken(data_EMTFShower_token_, dataShowers);
  e.getByToken(emul_EMTFShower_token_, emulShowers);

  for (auto dSh = dataShowers->begin(); dSh != dataShowers->end(); ++dSh) {
    if (dSh->isValid()) {
      emtfShowerDataSummary_denom_->Fill(dSh->sector(), (dSh->endcap() == 1) ? 1.5 : 0.5);
      for (auto eSh = emulShowers->begin(); eSh != emulShowers->end(); ++eSh) {
        if (eSh->isValid() and dSh->sector() == eSh->sector() and dSh->endcap() == eSh->endcap() and *dSh == *eSh)
          emtfShowerDataSummary_num_->Fill(dSh->sector(), (dSh->endcap() == 1) ? 1.5 : 0.5);
      }
    }
  }

  for (auto eSh = emulShowers->begin(); eSh != emulShowers->end(); ++eSh) {
    bool isMatched = false;
    if (eSh->isValid()) {
      emtfShowerEmulSummary_denom_->Fill(eSh->sector(), (eSh->endcap() == 1) ? 1.5 : 0.5);
      for (auto dSh = dataShowers->begin(); dSh != dataShowers->end(); ++dSh) {
        if (dSh->isValid() and eSh->sector() == dSh->sector() and eSh->endcap() == dSh->endcap() and *dSh == *eSh)
          isMatched = true;
      }
      if (not isMatched) {
        emtfShowerEmulSummary_num_->Fill(eSh->sector(), (eSh->endcap() == 1) ? 1.5 : 0.5);
      }
    }
  }
}
