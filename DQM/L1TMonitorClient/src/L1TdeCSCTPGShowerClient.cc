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

  lctShowerDataNomSummary_eff_ = iBooker.book2D("lct_cscshower_data_nom_summary_eff",
                                                "Efficiency of data LCT Nominal shower being correctly emulated",
                                                36,
                                                1,
                                                37,
                                                18,
                                                0,
                                                18);
  alctShowerDataNomSummary_eff_ = iBooker.book2D("alct_cscshower_data_nom_summary_eff",
                                                 "Efficiency of data ALCT Nominal shower being correctly emulated",
                                                 36,
                                                 1,
                                                 37,
                                                 18,
                                                 0,
                                                 18);
  clctShowerDataNomSummary_eff_ = iBooker.book2D("clct_cscshower_data_nom_summary_eff",
                                                 "Efficiency of data CLCT Nominal shower being correctly emulated",
                                                 36,
                                                 1,
                                                 37,
                                                 18,
                                                 0,
                                                 18);

  lctShowerEmulNomSummary_eff_ = iBooker.book2D("lct_cscshower_emul_nom_summary_eff",
                                                "Fraction of emulated LCT Nominal shower without matching data LCT",
                                                36,
                                                1,
                                                37,
                                                18,
                                                0,
                                                18);
  alctShowerEmulNomSummary_eff_ = iBooker.book2D("alct_cscshower_emul_nom_summary_eff",
                                                 "Fraction of emulated ALCT Nominal shower without matching data ALCT",
                                                 36,
                                                 1,
                                                 37,
                                                 18,
                                                 0,
                                                 18);
  clctShowerEmulNomSummary_eff_ = iBooker.book2D("clct_cscshower_emul_nom_summary_eff",
                                                 "Fraction of emulated CLCT Nominal shower without matching data CLCT",
                                                 36,
                                                 1,
                                                 37,
                                                 18,
                                                 0,
                                                 18);

  lctShowerDataTightSummary_eff_ = iBooker.book2D("lct_cscshower_data_tight_summary_eff",
                                                  "Efficiency of data LCT Tight shower being correctly emulated",
                                                  36,
                                                  1,
                                                  37,
                                                  18,
                                                  0,
                                                  18);
  alctShowerDataTightSummary_eff_ = iBooker.book2D("alct_cscshower_data_tight_summary_eff",
                                                   "Efficiency of data ALCT Tight shower being correctly emulated",
                                                   36,
                                                   1,
                                                   37,
                                                   18,
                                                   0,
                                                   18);
  clctShowerDataTightSummary_eff_ = iBooker.book2D("clct_cscshower_data_tight_summary_eff",
                                                   "Efficiency of data CLCT Tight shower being correctly emulated",
                                                   36,
                                                   1,
                                                   37,
                                                   18,
                                                   0,
                                                   18);

  lctShowerEmulTightSummary_eff_ = iBooker.book2D("lct_cscshower_emul_tight_summary_eff",
                                                  "Fraction of emulated LCT Tight shower without matching data LCT",
                                                  36,
                                                  1,
                                                  37,
                                                  18,
                                                  0,
                                                  18);
  alctShowerEmulTightSummary_eff_ = iBooker.book2D("alct_cscshower_emul_tight_summary_eff",
                                                   "Fraction of emulated ALCT Tight shower without matching data ALCT",
                                                   36,
                                                   1,
                                                   37,
                                                   18,
                                                   0,
                                                   18);
  clctShowerEmulTightSummary_eff_ = iBooker.book2D("clct_cscshower_emul_tight_summary_eff",
                                                   "Fraction of emulated CLCT Tight shower without matching data CLCT",
                                                   36,
                                                   1,
                                                   37,
                                                   18,
                                                   0,
                                                   18);

  // x labels
  lctShowerDataNomSummary_eff_->setAxisTitle("Chamber", 1);
  alctShowerDataNomSummary_eff_->setAxisTitle("Chamber", 1);
  clctShowerDataNomSummary_eff_->setAxisTitle("Chamber", 1);

  lctShowerEmulNomSummary_eff_->setAxisTitle("Chamber", 1);
  alctShowerEmulNomSummary_eff_->setAxisTitle("Chamber", 1);
  clctShowerEmulNomSummary_eff_->setAxisTitle("Chamber", 1);

  lctShowerDataTightSummary_eff_->setAxisTitle("Chamber", 1);
  alctShowerDataTightSummary_eff_->setAxisTitle("Chamber", 1);
  clctShowerDataTightSummary_eff_->setAxisTitle("Chamber", 1);

  lctShowerEmulTightSummary_eff_->setAxisTitle("Chamber", 1);
  alctShowerEmulTightSummary_eff_->setAxisTitle("Chamber", 1);
  clctShowerEmulTightSummary_eff_->setAxisTitle("Chamber", 1);

  // plotting option
  lctShowerDataNomSummary_eff_->setOption("colz");
  alctShowerDataNomSummary_eff_->setOption("colz");
  clctShowerDataNomSummary_eff_->setOption("colz");

  lctShowerEmulNomSummary_eff_->setOption("colz");
  alctShowerEmulNomSummary_eff_->setOption("colz");
  clctShowerEmulNomSummary_eff_->setOption("colz");

  lctShowerDataTightSummary_eff_->setOption("colz");
  alctShowerDataTightSummary_eff_->setOption("colz");
  clctShowerDataTightSummary_eff_->setOption("colz");

  lctShowerEmulTightSummary_eff_->setOption("colz");
  alctShowerEmulTightSummary_eff_->setOption("colz");
  clctShowerEmulTightSummary_eff_->setOption("colz");

  // summary plots
  const std::array<std::string, 9> suffix_label{{"4/2", "4/1", "3/2", "3/1", " 2/2", "2/1", "1/3", "1/2", "1/1"}};

  // y labels
  for (int ybin = 1; ybin <= 9; ++ybin) {
    lctShowerDataNomSummary_eff_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    alctShowerDataNomSummary_eff_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    clctShowerDataNomSummary_eff_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);

    lctShowerEmulNomSummary_eff_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    alctShowerEmulNomSummary_eff_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    clctShowerEmulNomSummary_eff_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);

    lctShowerDataNomSummary_eff_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    alctShowerDataNomSummary_eff_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    clctShowerDataNomSummary_eff_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);

    lctShowerEmulNomSummary_eff_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    alctShowerEmulNomSummary_eff_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    clctShowerEmulNomSummary_eff_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);

    lctShowerDataTightSummary_eff_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    alctShowerDataTightSummary_eff_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    clctShowerDataTightSummary_eff_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);

    lctShowerEmulTightSummary_eff_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    alctShowerEmulTightSummary_eff_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    clctShowerEmulTightSummary_eff_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);

    lctShowerDataTightSummary_eff_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    alctShowerDataTightSummary_eff_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    clctShowerDataTightSummary_eff_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);

    lctShowerEmulTightSummary_eff_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    alctShowerEmulTightSummary_eff_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    clctShowerEmulTightSummary_eff_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
  }
}

void L1TdeCSCTPGShowerClient::processHistograms(DQMStore::IGetter &igetter) {
  MonitorElement *lctShowerDataNomSummary_denom_ = igetter.get(monitorDir_ + "/lct_cscshower_data_nom_summary_denom");
  MonitorElement *lctShowerDataNomSummary_num_ = igetter.get(monitorDir_ + "/lct_cscshower_data_nom_summary_num");
  MonitorElement *alctShowerDataNomSummary_denom_ = igetter.get(monitorDir_ + "/alct_cscshower_data_nom_summary_denom");
  MonitorElement *alctShowerDataNomSummary_num_ = igetter.get(monitorDir_ + "/alct_cscshower_data_nom_summary_num");
  MonitorElement *clctShowerDataNomSummary_denom_ = igetter.get(monitorDir_ + "/clct_cscshower_data_nom_summary_denom");
  MonitorElement *clctShowerDataNomSummary_num_ = igetter.get(monitorDir_ + "/clct_cscshower_data_nom_summary_num");

  if (lctShowerDataNomSummary_denom_ == nullptr or lctShowerDataNomSummary_num_ == nullptr or
      alctShowerDataNomSummary_denom_ == nullptr or alctShowerDataNomSummary_num_ == nullptr or
      clctShowerDataNomSummary_denom_ == nullptr or clctShowerDataNomSummary_num_ == nullptr) {
    edm::LogWarning("L1TdeCSCTPGShowerClient")
        << __PRETTY_FUNCTION__ << " could not load the necessary shower data histograms for harvesting";
    return;
  }

  MonitorElement *lctShowerEmulNomSummary_denom_ = igetter.get(monitorDir_ + "/lct_cscshower_emul_nom_summary_denom");
  MonitorElement *lctShowerEmulNomSummary_num_ = igetter.get(monitorDir_ + "/lct_cscshower_emul_nom_summary_num");
  MonitorElement *alctShowerEmulNomSummary_denom_ = igetter.get(monitorDir_ + "/alct_cscshower_emul_nom_summary_denom");
  MonitorElement *alctShowerEmulNomSummary_num_ = igetter.get(monitorDir_ + "/alct_cscshower_emul_nom_summary_num");
  MonitorElement *clctShowerEmulNomSummary_denom_ = igetter.get(monitorDir_ + "/clct_cscshower_emul_nom_summary_denom");
  MonitorElement *clctShowerEmulNomSummary_num_ = igetter.get(monitorDir_ + "/clct_cscshower_emul_nom_summary_num");

  if (lctShowerEmulNomSummary_denom_ == nullptr or lctShowerEmulNomSummary_num_ == nullptr or
      alctShowerEmulNomSummary_denom_ == nullptr or alctShowerEmulNomSummary_num_ == nullptr or
      clctShowerEmulNomSummary_denom_ == nullptr or clctShowerEmulNomSummary_num_ == nullptr) {
    edm::LogWarning("L1TdeCSCTPGShowerClient")
        << __PRETTY_FUNCTION__ << " could not load the necessary shower emulation histograms for harvesting";
    return;
  }

  MonitorElement *lctShowerDataTightSummary_denom_ =
      igetter.get(monitorDir_ + "/lct_cscshower_data_tight_summary_denom");
  MonitorElement *lctShowerDataTightSummary_num_ = igetter.get(monitorDir_ + "/lct_cscshower_data_tight_summary_num");
  MonitorElement *alctShowerDataTightSummary_denom_ =
      igetter.get(monitorDir_ + "/alct_cscshower_data_tight_summary_denom");
  MonitorElement *alctShowerDataTightSummary_num_ = igetter.get(monitorDir_ + "/alct_cscshower_data_tight_summary_num");
  MonitorElement *clctShowerDataTightSummary_denom_ =
      igetter.get(monitorDir_ + "/clct_cscshower_data_tight_summary_denom");
  MonitorElement *clctShowerDataTightSummary_num_ = igetter.get(monitorDir_ + "/clct_cscshower_data_tight_summary_num");

  if (lctShowerDataTightSummary_denom_ == nullptr or lctShowerDataTightSummary_num_ == nullptr or
      alctShowerDataTightSummary_denom_ == nullptr or alctShowerDataTightSummary_num_ == nullptr or
      clctShowerDataTightSummary_denom_ == nullptr or clctShowerDataTightSummary_num_ == nullptr) {
    edm::LogWarning("L1TdeCSCTPGShowerClient")
        << __PRETTY_FUNCTION__ << " could not load the necessary shower data (tight) histograms for harvesting";
    return;
  }

  MonitorElement *lctShowerEmulTightSummary_denom_ =
      igetter.get(monitorDir_ + "/lct_cscshower_emul_tight_summary_denom");
  MonitorElement *lctShowerEmulTightSummary_num_ = igetter.get(monitorDir_ + "/lct_cscshower_emul_tight_summary_num");
  MonitorElement *alctShowerEmulTightSummary_denom_ =
      igetter.get(monitorDir_ + "/alct_cscshower_emul_tight_summary_denom");
  MonitorElement *alctShowerEmulTightSummary_num_ = igetter.get(monitorDir_ + "/alct_cscshower_emul_tight_summary_num");
  MonitorElement *clctShowerEmulTightSummary_denom_ =
      igetter.get(monitorDir_ + "/clct_cscshower_emul_tight_summary_denom");
  MonitorElement *clctShowerEmulTightSummary_num_ = igetter.get(monitorDir_ + "/clct_cscshower_emul_tight_summary_num");

  if (lctShowerEmulTightSummary_denom_ == nullptr or lctShowerEmulTightSummary_num_ == nullptr or
      alctShowerEmulTightSummary_denom_ == nullptr or alctShowerEmulTightSummary_num_ == nullptr or
      clctShowerEmulTightSummary_denom_ == nullptr or clctShowerEmulTightSummary_num_ == nullptr) {
    edm::LogWarning("L1TdeCSCTPGShowerClient")
        << __PRETTY_FUNCTION__ << " could not load the necessary shower emulation (tight) histograms for harvesting";
    return;
  }

  lctShowerDataNomSummary_eff_->getTH2F()->Divide(
      lctShowerDataNomSummary_num_->getTH2F(), lctShowerDataNomSummary_denom_->getTH2F(), 1, 1, "");
  alctShowerDataNomSummary_eff_->getTH2F()->Divide(
      alctShowerDataNomSummary_num_->getTH2F(), alctShowerDataNomSummary_denom_->getTH2F(), 1, 1, "");
  clctShowerDataNomSummary_eff_->getTH2F()->Divide(
      clctShowerDataNomSummary_num_->getTH2F(), clctShowerDataNomSummary_denom_->getTH2F(), 1, 1, "");

  lctShowerEmulNomSummary_eff_->getTH2F()->Divide(
      lctShowerEmulNomSummary_num_->getTH2F(), lctShowerEmulNomSummary_denom_->getTH2F(), 1, 1, "");
  alctShowerEmulNomSummary_eff_->getTH2F()->Divide(
      alctShowerEmulNomSummary_num_->getTH2F(), alctShowerEmulNomSummary_denom_->getTH2F(), 1, 1, "");
  clctShowerEmulNomSummary_eff_->getTH2F()->Divide(
      clctShowerEmulNomSummary_num_->getTH2F(), clctShowerEmulNomSummary_denom_->getTH2F(), 1, 1, "");

  lctShowerDataTightSummary_eff_->getTH2F()->Divide(
      lctShowerDataTightSummary_num_->getTH2F(), lctShowerDataTightSummary_denom_->getTH2F(), 1, 1, "");
  alctShowerDataTightSummary_eff_->getTH2F()->Divide(
      alctShowerDataTightSummary_num_->getTH2F(), alctShowerDataTightSummary_denom_->getTH2F(), 1, 1, "");
  clctShowerDataTightSummary_eff_->getTH2F()->Divide(
      clctShowerDataTightSummary_num_->getTH2F(), clctShowerDataTightSummary_denom_->getTH2F(), 1, 1, "");

  lctShowerEmulTightSummary_eff_->getTH2F()->Divide(
      lctShowerEmulTightSummary_num_->getTH2F(), lctShowerEmulTightSummary_denom_->getTH2F(), 1, 1, "");
  alctShowerEmulTightSummary_eff_->getTH2F()->Divide(
      alctShowerEmulTightSummary_num_->getTH2F(), alctShowerEmulTightSummary_denom_->getTH2F(), 1, 1, "");
  clctShowerEmulTightSummary_eff_->getTH2F()->Divide(
      clctShowerEmulTightSummary_num_->getTH2F(), clctShowerEmulTightSummary_denom_->getTH2F(), 1, 1, "");

  lctShowerDataNomSummary_eff_->getTH2F()->GetZaxis()->SetRangeUser(0.95, 1);
  alctShowerDataNomSummary_eff_->getTH2F()->GetZaxis()->SetRangeUser(0.95, 1);
  clctShowerDataNomSummary_eff_->getTH2F()->GetZaxis()->SetRangeUser(0.95, 1);

  lctShowerDataTightSummary_eff_->getTH2F()->GetZaxis()->SetRangeUser(0.95, 1);
  alctShowerDataTightSummary_eff_->getTH2F()->GetZaxis()->SetRangeUser(0.95, 1);
  clctShowerDataTightSummary_eff_->getTH2F()->GetZaxis()->SetRangeUser(0.95, 1);
}
