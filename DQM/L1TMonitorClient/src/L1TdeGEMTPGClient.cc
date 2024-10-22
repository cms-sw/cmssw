#include "DQM/L1TMonitorClient/interface/L1TdeGEMTPGClient.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "TRandom.h"
using namespace edm;
using namespace std;

L1TdeGEMTPGClient::L1TdeGEMTPGClient(const edm::ParameterSet &ps)
    : monitorDir_(ps.getParameter<string>("monitorDir")),
      chambers_(ps.getParameter<std::vector<std::string>>("chambers")),
      // variables
      clusterVars_(ps.getParameter<std::vector<std::string>>("clusterVars")),
      // binning
      clusterNBin_(ps.getParameter<std::vector<unsigned>>("clusterNBin")),
      clusterMinBin_(ps.getParameter<std::vector<double>>("clusterMinBin")),
      clusterMaxBin_(ps.getParameter<std::vector<double>>("clusterMaxBin")) {}

L1TdeGEMTPGClient::~L1TdeGEMTPGClient() {}

void L1TdeGEMTPGClient::dqmEndLuminosityBlock(DQMStore::IBooker &ibooker,
                                              DQMStore::IGetter &igetter,
                                              const edm::LuminosityBlock &lumiSeg,
                                              const edm::EventSetup &c) {
  book(ibooker);
  processHistograms(igetter);
}

//--------------------------------------------------------
void L1TdeGEMTPGClient::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  book(ibooker);
  processHistograms(igetter);
}

void L1TdeGEMTPGClient::book(DQMStore::IBooker &iBooker) {
  iBooker.setCurrentFolder(monitorDir_);

  // chamber type
  for (unsigned iType = 0; iType < chambers_.size(); iType++) {
    // cluster variable
    for (unsigned iVar = 0; iVar < clusterVars_.size(); iVar++) {
      const std::string key("cluster_" + clusterVars_[iVar] + "_diff");
      const std::string histName(key + "_" + chambers_[iType]);
      const std::string histTitle(chambers_[iType] + " Cluster " + clusterVars_[iVar] + " (Emul - Data)");
      if (chamberHistos_[iType][key] == nullptr)
        chamberHistos_[iType][key] =
            iBooker.book1D(histName, histTitle, clusterNBin_[iVar], clusterMinBin_[iVar], clusterMaxBin_[iVar]);
      else
        chamberHistos_[iType][key]->Reset();
    }
  }
}

void L1TdeGEMTPGClient::processHistograms(DQMStore::IGetter &igetter) {
  MonitorElement *dataMon;
  MonitorElement *emulMon;

  // chamber type
  for (unsigned iType = 0; iType < chambers_.size(); iType++) {
    // cluster variable
    for (unsigned iVar = 0; iVar < clusterVars_.size(); iVar++) {
      const std::string key("cluster_" + clusterVars_[iVar]);
      const std::string histData(key + "_data_" + chambers_[iType]);
      const std::string histEmul(key + "_emul_" + chambers_[iType]);

      dataMon = igetter.get(monitorDir_ + "/" + histData);
      emulMon = igetter.get(monitorDir_ + "/" + histEmul);

      TH1F *hDiff = chamberHistos_[iType][key + "_diff"]->getTH1F();

      if (dataMon && emulMon) {
        TH1F *dataHist = dataMon->getTH1F();
        TH1F *emulHist = emulMon->getTH1F();
        hDiff->Add(emulHist, dataHist, 1, -1);
      }
    }
  }
}
