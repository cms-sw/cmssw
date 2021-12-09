#include "DQM/L1TMonitorClient/interface/L1TdeCSCTPGShowerClient.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "TRandom.h"
using namespace edm;
using namespace std;

L1TdeCSCTPGShowerClient::L1TdeCSCTPGShowerClient(const edm::ParameterSet &ps)
    : monitorDir_(ps.getUntrackedParameter<string>("monitorDir")) {}

L1TdeCSCTPGShowerClient::~L1TdeCSCTPGShowerClient() {}

void L1TdeCSCTPGShowerClient::dqmEndLuminosityBlock(DQMStore::IBooker &ibooker,
                                                    DQMStore::IGetter &igetter,
                                                    const edm::LuminosityBlock &lumiSeg,
                                                    const edm::EventSetup &c) {
  book(ibooker);
  processHistograms(igetter);
}

//--------------------------------------------------------
void L1TdeCSCTPGShowerClient::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  book(ibooker);
  processHistograms(igetter);
}

void L1TdeCSCTPGShowerClient::book(DQMStore::IBooker &iBooker) {
  iBooker.setCurrentFolder(monitorDir_);

  lctShowerDataSummary_eff_ = iBooker.book2D(
      "lct_cscshower_data_summary_eff", "Efficiency of data LCT shower being correctly emulated", 36, 1, 37, 18, 0, 18);
  alctShowerDataSummary_eff_ = iBooker.book2D("alct_cscshower_data_summary_eff",
                                              "Efficiency of data ALCT shower being correctly emulated",
                                              36,
                                              1,
                                              37,
                                              18,
                                              0,
                                              18);
  clctShowerDataSummary_eff_ = iBooker.book2D("clct_cscshower_data_summary_eff",
                                              "Efficiency of data CLCT shower being correctly emulated",
                                              36,
                                              1,
                                              37,
                                              18,
                                              0,
                                              18);

  lctShowerEmulSummary_eff_ = iBooker.book2D("lct_cscshower_emul_summary_eff",
                                             "Fraction of emulated LCT shower without matching data LCT",
                                             36,
                                             1,
                                             37,
                                             18,
                                             0,
                                             18);
  alctShowerEmulSummary_eff_ = iBooker.book2D("alct_cscshower_emul_summary_eff",
                                              "Fraction of emulated ALCT shower without matching data ALCT",
                                              36,
                                              1,
                                              37,
                                              18,
                                              0,
                                              18);
  clctShowerEmulSummary_eff_ = iBooker.book2D("clct_cscshower_emul_summary_eff",
                                              "Fraction of emulated CLCT shower without matching data CLCT",
                                              36,
                                              1,
                                              37,
                                              18,
                                              0,
                                              18);

  // x labels
  lctShowerDataSummary_eff_->setAxisTitle("Chamber", 1);
  alctShowerDataSummary_eff_->setAxisTitle("Chamber", 1);
  clctShowerDataSummary_eff_->setAxisTitle("Chamber", 1);

  lctShowerEmulSummary_eff_->setAxisTitle("Chamber", 1);
  alctShowerEmulSummary_eff_->setAxisTitle("Chamber", 1);
  clctShowerEmulSummary_eff_->setAxisTitle("Chamber", 1);

  // plotting option
  lctShowerDataSummary_eff_->setOption("colz");
  alctShowerDataSummary_eff_->setOption("colz");
  clctShowerDataSummary_eff_->setOption("colz");

  lctShowerEmulSummary_eff_->setOption("colz");
  alctShowerEmulSummary_eff_->setOption("colz");
  clctShowerEmulSummary_eff_->setOption("colz");

  // summary plots
  const std::array<std::string, 9> suffix_label{{"4/2", "4/1", "3/2", "3/1", " 2/2", "2/1", "1/3", "1/2", "1/1"}};

  // y labels
  for (int ybin = 1; ybin <= 9; ++ybin) {
    lctShowerDataSummary_eff_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    alctShowerDataSummary_eff_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    clctShowerDataSummary_eff_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);

    lctShowerEmulSummary_eff_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    alctShowerEmulSummary_eff_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    clctShowerEmulSummary_eff_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);

    lctShowerDataSummary_eff_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    alctShowerDataSummary_eff_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    clctShowerDataSummary_eff_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);

    lctShowerEmulSummary_eff_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    alctShowerEmulSummary_eff_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    clctShowerEmulSummary_eff_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
  }
}

void L1TdeCSCTPGShowerClient::processHistograms(DQMStore::IGetter &igetter) {
  MonitorElement *lctShowerDataSummary_denom_ = igetter.get(monitorDir_ + "/lct_cscshower_data_summary_denom");
  MonitorElement *lctShowerDataSummary_num_ = igetter.get(monitorDir_ + "/lct_cscshower_data_summary_num");
  MonitorElement *alctShowerDataSummary_denom_ = igetter.get(monitorDir_ + "/alct_cscshower_data_summary_denom");
  MonitorElement *alctShowerDataSummary_num_ = igetter.get(monitorDir_ + "/alct_cscshower_data_summary_num");
  MonitorElement *clctShowerDataSummary_denom_ = igetter.get(monitorDir_ + "/clct_cscshower_data_summary_denom");
  MonitorElement *clctShowerDataSummary_num_ = igetter.get(monitorDir_ + "/clct_cscshower_data_summary_num");

  MonitorElement *lctShowerEmulSummary_denom_ = igetter.get(monitorDir_ + "/lct_cscshower_emul_summary_denom");
  MonitorElement *lctShowerEmulSummary_num_ = igetter.get(monitorDir_ + "/lct_cscshower_emul_summary_num");
  MonitorElement *alctShowerEmulSummary_denom_ = igetter.get(monitorDir_ + "/alct_cscshower_emul_summary_denom");
  MonitorElement *alctShowerEmulSummary_num_ = igetter.get(monitorDir_ + "/alct_cscshower_emul_summary_num");
  MonitorElement *clctShowerEmulSummary_denom_ = igetter.get(monitorDir_ + "/clct_cscshower_emul_summary_denom");
  MonitorElement *clctShowerEmulSummary_num_ = igetter.get(monitorDir_ + "/clct_cscshower_emul_summary_num");

  lctShowerDataSummary_eff_->getTH2F()->Divide(
      lctShowerDataSummary_num_->getTH2F(), lctShowerDataSummary_denom_->getTH2F(), 1, 1, "");
  alctShowerDataSummary_eff_->getTH2F()->Divide(
      alctShowerDataSummary_num_->getTH2F(), alctShowerDataSummary_denom_->getTH2F(), 1, 1, "");
  clctShowerDataSummary_eff_->getTH2F()->Divide(
      clctShowerDataSummary_num_->getTH2F(), clctShowerDataSummary_denom_->getTH2F(), 1, 1, "");

  lctShowerEmulSummary_eff_->getTH2F()->Divide(
      lctShowerEmulSummary_num_->getTH2F(), lctShowerEmulSummary_denom_->getTH2F(), 1, 1, "");
  alctShowerEmulSummary_eff_->getTH2F()->Divide(
      alctShowerEmulSummary_num_->getTH2F(), alctShowerEmulSummary_denom_->getTH2F(), 1, 1, "");
  clctShowerEmulSummary_eff_->getTH2F()->Divide(
      clctShowerEmulSummary_num_->getTH2F(), clctShowerEmulSummary_denom_->getTH2F(), 1, 1, "");

  lctShowerDataSummary_eff_->getTH2F()->GetZaxis()->SetRangeUser(0.95, 1);
  alctShowerDataSummary_eff_->getTH2F()->GetZaxis()->SetRangeUser(0.95, 1);
  clctShowerDataSummary_eff_->getTH2F()->GetZaxis()->SetRangeUser(0.95, 1);
}
