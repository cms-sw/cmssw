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
      lctMaxBin_(ps.getParameter<std::vector<double>>("lctMaxBin")) {}

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

void L1TdeCSCTPGClient::processHistograms(DQMStore::IGetter &igetter) {
  MonitorElement *dataMon;
  MonitorElement *emulMon;

  // chamber type
  for (unsigned iType = 0; iType < chambers_.size(); iType++) {
    // alct variable
    for (unsigned iVar = 0; iVar < alctVars_.size(); iVar++) {
      const std::string key("alct_" + alctVars_[iVar]);
      const std::string histData(key + "_data_" + chambers_[iType]);
      const std::string histEmul(key + "_emul_" + chambers_[iType]);

      dataMon = igetter.get(monitorDir_ + "/" + histData);
      emulMon = igetter.get(monitorDir_ + "/" + histEmul);

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

      TH1F *hDiff = chamberHistos_[iType][key + "_diff"]->getTH1F();

      if (dataMon && emulMon) {
        TH1F *hData = dataMon->getTH1F();
        TH1F *hEmul = emulMon->getTH1F();
        hDiff->Add(hEmul, hData, 1, -1);
      }
    }
  }
}
