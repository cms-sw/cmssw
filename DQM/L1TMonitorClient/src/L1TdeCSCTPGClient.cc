#include "DQM/L1TMonitorClient/interface/L1TdeCSCTPGClient.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "TRandom.h"
using namespace edm;
using namespace std;

L1TdeCSCTPGClient::L1TdeCSCTPGClient(const edm::ParameterSet &ps)
    : monitorDir_(ps.getParameter<string>("monitorDir")),
      chambers_(ps.getParameter<std::vector<std::string>>("chambers")),
      // variables
      alctVars_(ps.getParameter<std::vector<std::string>>("alctVars")),
      clctVars_(ps.getParameter<std::vector<std::string>>("clctVars")),
      lctVars_(ps.getParameter<std::vector<std::string>>("lctVars")),
      // binning
      alctNBin_(ps.getParameter<std::vector<unsigned>>("alctNBin")),
      clctNBin_(ps.getParameter<std::vector<unsigned>>("clctNBin")),
      lctNBin_(ps.getParameter<std::vector<unsigned>>("lctNBin")),
      alctMinBin_(ps.getParameter<std::vector<double>>("alctMinBin")),
      clctMinBin_(ps.getParameter<std::vector<double>>("clctMinBin")),
      lctMinBin_(ps.getParameter<std::vector<double>>("lctMinBin")),
      alctMaxBin_(ps.getParameter<std::vector<double>>("alctMaxBin")),
      clctMaxBin_(ps.getParameter<std::vector<double>>("clctMaxBin")),
      lctMaxBin_(ps.getParameter<std::vector<double>>("lctMaxBin")),
      useB904ME11_(ps.getParameter<bool>("useB904ME11")),
      useB904ME21_(ps.getParameter<bool>("useB904ME21")),
      useB904ME234s2_(ps.getParameter<bool>("useB904ME234s2")),
      isRun3_(ps.getParameter<bool>("isRun3")),
      // by default the DQM will make 2D summary plots. Do you also want
      // the very large number of 1D plots? Typically only for testing at B904 or
      // on select P5 data
      make1DPlots_(ps.getParameter<bool>("make1DPlots")) {
  useB904_ = useB904ME11_ or useB904ME21_ or useB904ME234s2_;
}

L1TdeCSCTPGClient::~L1TdeCSCTPGClient() {}

void L1TdeCSCTPGClient::dqmEndLuminosityBlock(DQMStore::IBooker &ibooker,
                                              DQMStore::IGetter &igetter,
                                              const edm::LuminosityBlock &lumiSeg,
                                              const edm::EventSetup &c) {
  book(ibooker);
  processHistograms(igetter);
}

//--------------------------------------------------------
void L1TdeCSCTPGClient::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  book(ibooker);
  processHistograms(igetter);
}

void L1TdeCSCTPGClient::book(DQMStore::IBooker &iBooker) {
  iBooker.setCurrentFolder(monitorDir_);

  // remove the non-ME1/1 chambers from the list when useB904ME11 is set to true
  if (useB904ME11_) {
    chambers_.resize(1);
  }
  // similar for ME2/1
  else if (useB904ME21_) {
    auto temp = chambers_[3];
    chambers_.resize(1);
    chambers_[0] = temp;
  }
  // similar for ME4/2
  else if (useB904ME234s2_) {
    auto temp = chambers_.back();
    chambers_.resize(1);
    chambers_[0] = temp;
  }
  // collision data in Run-3
  else if (isRun3_) {
    clctVars_.resize(9);
    lctVars_.resize(9);
  }
  // do not analyze Run-3 properties in Run-1 and Run-2 eras
  else {
    clctVars_.resize(4);
    lctVars_.resize(5);
  }

  // 1D plots for experts
  if (useB904ME11_ or useB904ME21_ or useB904ME234s2_ or make1DPlots_) {
    // chamber type
    for (unsigned iType = 0; iType < chambers_.size(); iType++) {
      // alct variable
      for (unsigned iVar = 0; iVar < alctVars_.size(); iVar++) {
        const std::string key("alct_" + alctVars_[iVar] + "_diff");
        const std::string histName(key + "_" + chambers_[iType]);
        const std::string histTitle(chambers_[iType] + " ALCT " + alctVars_[iVar] + " (Emul - Data)");
        if (chamberHistos_[iType][key] == nullptr)
          chamberHistos_[iType][key] =
              iBooker.book1D(histName, histTitle, alctNBin_[iVar], alctMinBin_[iVar], alctMaxBin_[iVar]);
        else
          chamberHistos_[iType][key]->Reset();
      }

      // clct variable
      for (unsigned iVar = 0; iVar < clctVars_.size(); iVar++) {
        const std::string key("clct_" + clctVars_[iVar] + "_diff");
        const std::string histName(key + "_" + chambers_[iType]);
        const std::string histTitle(chambers_[iType] + " CLCT " + clctVars_[iVar] + " (Emul - Data)");
        if (chamberHistos_[iType][key] == nullptr)
          chamberHistos_[iType][key] =
              iBooker.book1D(histName, histTitle, clctNBin_[iVar], clctMinBin_[iVar], clctMaxBin_[iVar]);
        else
          chamberHistos_[iType][key]->Reset();
      }

      // lct variable
      for (unsigned iVar = 0; iVar < lctVars_.size(); iVar++) {
        const std::string key("lct_" + lctVars_[iVar] + "_diff");
        const std::string histName(key + "_" + chambers_[iType]);
        const std::string histTitle(chambers_[iType] + " LCT " + lctVars_[iVar] + " (Emul - Data)");
        if (chamberHistos_[iType][key] == nullptr)
          chamberHistos_[iType][key] =
              iBooker.book1D(histName, histTitle, lctNBin_[iVar], lctMinBin_[iVar], lctMaxBin_[iVar]);
        else
          chamberHistos_[iType][key]->Reset();
      }
    }
  }

  // 2D summary plots
  lctDataSummary_eff_ = iBooker.book2D(
      "lct_csctp_data_summary_eff", "Efficiency of data LCT being correctly emulated", 36, 1, 37, 18, 0, 18);
  alctDataSummary_eff_ = iBooker.book2D(
      "alct_csctp_data_summary_eff", "Efficiency of data ALCT being correctly emulated", 36, 1, 37, 18, 0, 18);
  clctDataSummary_eff_ = iBooker.book2D(
      "clct_csctp_data_summary_eff", "Efficiency of data CLCT being correctly emulated", 36, 1, 37, 18, 0, 18);

  lctEmulSummary_eff_ = iBooker.book2D(
      "lct_csctp_emul_summary_eff", "Fraction of emulated LCT without matching data LCT", 36, 1, 37, 18, 0, 18);
  alctEmulSummary_eff_ = iBooker.book2D(
      "alct_csctp_emul_summary_eff", "Fraction of emulated ALCT without matching data ALCT", 36, 1, 37, 18, 0, 18);
  clctEmulSummary_eff_ = iBooker.book2D(
      "clct_csctp_emul_summary_eff", "Fraction of emulated CLCT without matching data CLCT", 36, 1, 37, 18, 0, 18);

  // x labels
  lctDataSummary_eff_->setAxisTitle("Chamber", 1);
  alctDataSummary_eff_->setAxisTitle("Chamber", 1);
  clctDataSummary_eff_->setAxisTitle("Chamber", 1);

  lctEmulSummary_eff_->setAxisTitle("Chamber", 1);
  alctEmulSummary_eff_->setAxisTitle("Chamber", 1);
  clctEmulSummary_eff_->setAxisTitle("Chamber", 1);

  // plotting option
  lctDataSummary_eff_->setOption("colz");
  alctDataSummary_eff_->setOption("colz");
  clctDataSummary_eff_->setOption("colz");

  lctEmulSummary_eff_->setOption("colz");
  alctEmulSummary_eff_->setOption("colz");
  clctEmulSummary_eff_->setOption("colz");

  // summary plots
  const std::array<std::string, 9> suffix_label{{"4/2", "4/1", "3/2", "3/1", " 2/2", "2/1", "1/3", "1/2", "1/1"}};

  // y labels
  for (int ybin = 1; ybin <= 9; ++ybin) {
    lctDataSummary_eff_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    alctDataSummary_eff_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    clctDataSummary_eff_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);

    lctEmulSummary_eff_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    alctEmulSummary_eff_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    clctEmulSummary_eff_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);

    lctDataSummary_eff_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    alctDataSummary_eff_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    clctDataSummary_eff_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);

    lctEmulSummary_eff_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    alctEmulSummary_eff_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    clctEmulSummary_eff_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
  }
}

void L1TdeCSCTPGClient::processHistograms(DQMStore::IGetter &igetter) {
  MonitorElement *dataMon;
  MonitorElement *emulMon;

  // 1D plots for experts
  if (useB904ME11_ or useB904ME21_ or useB904ME234s2_ or make1DPlots_) {
    // chamber type
    for (unsigned iType = 0; iType < chambers_.size(); iType++) {
      // alct variable
      for (unsigned iVar = 0; iVar < alctVars_.size(); iVar++) {
        const std::string key("alct_" + alctVars_[iVar]);
        const std::string histData(key + "_data_" + chambers_[iType]);
        const std::string histEmul(key + "_emul_" + chambers_[iType]);

        dataMon = igetter.get(monitorDir_ + "/" + histData);
        emulMon = igetter.get(monitorDir_ + "/" + histEmul);

        if (dataMon == nullptr or emulMon == nullptr) {
          edm::LogWarning("L1TdeCSCTPGClient")
              << __PRETTY_FUNCTION__ << " could not load the necessary histograms for harvesting " << histData << " / "
              << histEmul;
          continue;
        }

        TH1F *hDiff = chamberHistos_[iType][key + "_diff"]->getTH1F();

        if (dataMon && emulMon) {
          TH1F *hData = dataMon->getTH1F();
          TH1F *hEmul = emulMon->getTH1F();
          hDiff->Add(hEmul, hData, 1, -1);
        }
      }

      // clct variable
      for (unsigned iVar = 0; iVar < clctVars_.size(); iVar++) {
        const std::string key("clct_" + clctVars_[iVar]);
        const std::string histData(key + "_data_" + chambers_[iType]);
        const std::string histEmul(key + "_emul_" + chambers_[iType]);

        dataMon = igetter.get(monitorDir_ + "/" + histData);
        emulMon = igetter.get(monitorDir_ + "/" + histEmul);

        if (dataMon == nullptr or emulMon == nullptr) {
          edm::LogWarning("L1TdeCSCTPGClient")
              << __PRETTY_FUNCTION__ << " could not load the necessary histograms for harvesting " << histData << " / "
              << histEmul;
          continue;
        }

        TH1F *hDiff = chamberHistos_[iType][key + "_diff"]->getTH1F();

        if (dataMon && emulMon) {
          TH1F *hData = dataMon->getTH1F();
          TH1F *hEmul = emulMon->getTH1F();
          hDiff->Add(hEmul, hData, 1, -1);
        }
      }

      // lct variable
      for (unsigned iVar = 0; iVar < lctVars_.size(); iVar++) {
        const std::string key("lct_" + lctVars_[iVar]);
        const std::string histData(key + "_data_" + chambers_[iType]);
        const std::string histEmul(key + "_emul_" + chambers_[iType]);

        dataMon = igetter.get(monitorDir_ + "/" + histData);
        emulMon = igetter.get(monitorDir_ + "/" + histEmul);

        if (dataMon == nullptr or emulMon == nullptr) {
          edm::LogWarning("L1TdeCSCTPGClient")
              << __PRETTY_FUNCTION__ << " could not load the necessary histograms for harvesting " << histData << " / "
              << histEmul;
          continue;
        }

        TH1F *hDiff = chamberHistos_[iType][key + "_diff"]->getTH1F();

        if (dataMon && emulMon) {
          TH1F *hData = dataMon->getTH1F();
          TH1F *hEmul = emulMon->getTH1F();
          hDiff->Add(hEmul, hData, 1, -1);
        }
      }
    }
  }

  // 2D summary plot
  MonitorElement *lctDataSummary_denom_ = igetter.get(monitorDir_ + "/lct_csctp_data_summary_denom");
  MonitorElement *lctDataSummary_num_ = igetter.get(monitorDir_ + "/lct_csctp_data_summary_num");
  MonitorElement *alctDataSummary_denom_ = igetter.get(monitorDir_ + "/alct_csctp_data_summary_denom");
  MonitorElement *alctDataSummary_num_ = igetter.get(monitorDir_ + "/alct_csctp_data_summary_num");
  MonitorElement *clctDataSummary_denom_ = igetter.get(monitorDir_ + "/clct_csctp_data_summary_denom");
  MonitorElement *clctDataSummary_num_ = igetter.get(monitorDir_ + "/clct_csctp_data_summary_num");

  if (lctDataSummary_denom_ == nullptr or lctDataSummary_num_ == nullptr or alctDataSummary_denom_ == nullptr or
      alctDataSummary_num_ == nullptr or clctDataSummary_denom_ == nullptr or clctDataSummary_num_ == nullptr) {
    edm::LogWarning("L1TdeCSCTPGClient") << __PRETTY_FUNCTION__
                                         << " could not load the necessary data histograms for 2D summary plots";
    return;
  }

  MonitorElement *lctEmulSummary_denom_ = igetter.get(monitorDir_ + "/lct_csctp_emul_summary_denom");
  MonitorElement *lctEmulSummary_num_ = igetter.get(monitorDir_ + "/lct_csctp_emul_summary_num");
  MonitorElement *alctEmulSummary_denom_ = igetter.get(monitorDir_ + "/alct_csctp_emul_summary_denom");
  MonitorElement *alctEmulSummary_num_ = igetter.get(monitorDir_ + "/alct_csctp_emul_summary_num");
  MonitorElement *clctEmulSummary_denom_ = igetter.get(monitorDir_ + "/clct_csctp_emul_summary_denom");
  MonitorElement *clctEmulSummary_num_ = igetter.get(monitorDir_ + "/clct_csctp_emul_summary_num");

  if (lctEmulSummary_denom_ == nullptr or lctEmulSummary_num_ == nullptr or alctEmulSummary_denom_ == nullptr or
      alctEmulSummary_num_ == nullptr or clctEmulSummary_denom_ == nullptr or clctEmulSummary_num_ == nullptr) {
    edm::LogWarning("L1TdeCSCTPGClient")
        << __PRETTY_FUNCTION__ << " could not load the necessary emulation histograms for the 2D summary plots";
    return;
  }

  lctDataSummary_eff_->getTH2F()->Divide(lctDataSummary_num_->getTH2F(), lctDataSummary_denom_->getTH2F(), 1, 1, "");
  alctDataSummary_eff_->getTH2F()->Divide(alctDataSummary_num_->getTH2F(), alctDataSummary_denom_->getTH2F(), 1, 1, "");
  clctDataSummary_eff_->getTH2F()->Divide(clctDataSummary_num_->getTH2F(), clctDataSummary_denom_->getTH2F(), 1, 1, "");

  lctEmulSummary_eff_->getTH2F()->Divide(lctEmulSummary_num_->getTH2F(), lctEmulSummary_denom_->getTH2F(), 1, 1, "");
  alctEmulSummary_eff_->getTH2F()->Divide(alctEmulSummary_num_->getTH2F(), alctEmulSummary_denom_->getTH2F(), 1, 1, "");
  clctEmulSummary_eff_->getTH2F()->Divide(clctEmulSummary_num_->getTH2F(), clctEmulSummary_denom_->getTH2F(), 1, 1, "");

  // set minima to 0.95 so the contrast comes out better!
  lctDataSummary_eff_->getTH2F()->GetZaxis()->SetRangeUser(0.95, 1);
  alctDataSummary_eff_->getTH2F()->GetZaxis()->SetRangeUser(0.95, 1);
  clctDataSummary_eff_->getTH2F()->GetZaxis()->SetRangeUser(0.95, 1);
}
