#include <string>

#include "DQM/L1TMonitor/interface/L1TdeStage2RegionalShower.h"

L1TdeStage2RegionalShower::L1TdeStage2RegionalShower(const edm::ParameterSet& ps)
    : data_EMTFShower_token_(
          consumes<l1t::RegionalMuonShowerBxCollection>(ps.getParameter<edm::InputTag>("dataSource"))),
      emul_EMTFShower_token_(
          consumes<l1t::RegionalMuonShowerBxCollection>(ps.getParameter<edm::InputTag>("emulSource"))),
      monitorDir_(ps.getUntrackedParameter<std::string>("monitorDir")) {}

L1TdeStage2RegionalShower::~L1TdeStage2RegionalShower() {}

void L1TdeStage2RegionalShower::bookHistograms(DQMStore::IBooker& iBooker, const edm::Run&, const edm::EventSetup&) {
  iBooker.setCurrentFolder(monitorDir_);

  // 2D summary plots
  emtfShowerDataSummary_denom_ =
      iBooker.book2D("emtf_shower_data_summary_denom", "Data EMTF Shower All", 6, 1, 7, 4, 0, 4);
  emtfShowerDataSummary_num_ =
      iBooker.book2D("emtf_shower_data_summary_num", "Data EMTF Shower Emul Matched", 6, 1, 7, 4, 0, 4);
  emtfShowerEmulSummary_denom_ =
      iBooker.book2D("emtf_shower_emul_summary_denom", "Emul EMTF Shower All", 6, 1, 7, 4, 0, 4);
  emtfShowerEmulSummary_num_ =
      iBooker.book2D("emtf_shower_emul_summary_num", "Emul EMTF Shower Not Matched to Data", 6, 1, 7, 4, 0, 4);

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
  emtfShowerDataSummary_denom_->setBinLabel(1, "ME- Tight", 2);
  emtfShowerDataSummary_num_->setBinLabel(1, "ME- Tight", 2);
  emtfShowerEmulSummary_denom_->setBinLabel(1, "ME- Tight", 2);
  emtfShowerEmulSummary_num_->setBinLabel(1, "ME- Tight", 2);
  emtfShowerDataSummary_denom_->setBinLabel(2, "ME- Nom", 2);
  emtfShowerDataSummary_num_->setBinLabel(2, "ME- Nom", 2);
  emtfShowerEmulSummary_denom_->setBinLabel(2, "ME- Nom", 2);
  emtfShowerEmulSummary_num_->setBinLabel(2, "ME- Nom", 2);

  emtfShowerDataSummary_denom_->setBinLabel(3, "ME+ Nom", 2);
  emtfShowerDataSummary_num_->setBinLabel(3, "ME+ Nom", 2);
  emtfShowerEmulSummary_denom_->setBinLabel(3, "ME+ Nom", 2);
  emtfShowerEmulSummary_num_->setBinLabel(3, "ME+ Nom", 2);
  emtfShowerDataSummary_denom_->setBinLabel(4, "ME+ Tight", 2);
  emtfShowerDataSummary_num_->setBinLabel(4, "ME+ Tight", 2);
  emtfShowerEmulSummary_denom_->setBinLabel(4, "ME+ Tight", 2);
  emtfShowerEmulSummary_num_->setBinLabel(4, "ME+ Tight", 2);
}

void L1TdeStage2RegionalShower::analyze(const edm::Event& e, const edm::EventSetup& c) {
  edm::Handle<l1t::RegionalMuonShowerBxCollection> dataShowers;
  edm::Handle<l1t::RegionalMuonShowerBxCollection> emulShowers;

  e.getByToken(data_EMTFShower_token_, dataShowers);
  e.getByToken(emul_EMTFShower_token_, emulShowers);

  for (auto dSh = dataShowers->begin(); dSh != dataShowers->end(); ++dSh) {
    if (dSh->isValid() and dSh->isOneNominalInTime()) {
      if (dSh->isOneTightInTime())
        emtfShowerDataSummary_denom_->Fill(dSh->processor() + 1,
                                           (dSh->trackFinderType() == l1t::tftype::emtf_pos) ? 3.5 : 0.5);
      emtfShowerDataSummary_denom_->Fill(dSh->processor() + 1,
                                         (dSh->trackFinderType() == l1t::tftype::emtf_pos) ? 2.5 : 1.5);
      for (auto eSh = emulShowers->begin(); eSh != emulShowers->end(); ++eSh) {
        if (eSh->isValid() and eSh->isOneNominalInTime() and dSh->processor() == eSh->processor() and
            dSh->trackFinderType() == eSh->trackFinderType() and *dSh == *eSh) {
          if (dSh->isOneTightInTime())
            emtfShowerDataSummary_num_->Fill(dSh->processor() + 1,
                                             (dSh->trackFinderType() == l1t::tftype::emtf_pos) ? 3.5 : 0.5);
          emtfShowerDataSummary_num_->Fill(dSh->processor() + 1,
                                           (dSh->trackFinderType() == l1t::tftype::emtf_pos) ? 2.5 : 1.5);
        }
      }
    }
  }

  for (auto eSh = emulShowers->begin(); eSh != emulShowers->end(); ++eSh) {
    bool isMatched = false;
    if (eSh->isValid() and eSh->isOneNominalInTime()) {
      if (eSh->isOneTightInTime())
        emtfShowerEmulSummary_denom_->Fill(eSh->processor() + 1,
                                           (eSh->trackFinderType() == l1t::tftype::emtf_pos) ? 3.5 : 0.5);
      emtfShowerEmulSummary_denom_->Fill(eSh->processor() + 1,
                                         (eSh->trackFinderType() == l1t::tftype::emtf_pos) ? 2.5 : 1.5);
      for (auto dSh = dataShowers->begin(); dSh != dataShowers->end(); ++dSh) {
        if (dSh->isValid() and dSh->isOneNominalInTime() and eSh->processor() == dSh->processor() and
            eSh->trackFinderType() == dSh->trackFinderType() and *dSh == *eSh)
          isMatched = true;
      }
      if (not isMatched) {
        if (eSh->isOneTightInTime())
          emtfShowerEmulSummary_num_->Fill(eSh->processor() + 1,
                                           (eSh->trackFinderType() == l1t::tftype::emtf_pos) ? 3.5 : 0.5);
        emtfShowerEmulSummary_num_->Fill(eSh->processor() + 1,
                                         (eSh->trackFinderType() == l1t::tftype::emtf_pos) ? 2.5 : 1.5);
      }
    }
  }
}
