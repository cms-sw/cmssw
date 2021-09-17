#include <string>

#include "DQM/L1TMonitor/interface/L1TdeGEMTPG.h"

L1TdeGEMTPG::L1TdeGEMTPG(const edm::ParameterSet& ps)
    : data_token_(consumes<GEMPadDigiClusterCollection>(ps.getParameter<edm::InputTag>("data"))),
      emul_token_(consumes<GEMPadDigiClusterCollection>(ps.getParameter<edm::InputTag>("emul"))),
      monitorDir_(ps.getParameter<std::string>("monitorDir")),
      verbose_(ps.getParameter<bool>("verbose")),

      chambers_(ps.getParameter<std::vector<std::string>>("chambers")),
      dataEmul_(ps.getParameter<std::vector<std::string>>("dataEmul")),

      // variables
      clusterVars_(ps.getParameter<std::vector<std::string>>("clusterVars")),

      // binning
      clusterNBin_(ps.getParameter<std::vector<unsigned>>("clusterNBin")),
      clusterMinBin_(ps.getParameter<std::vector<double>>("clusterMinBin")),
      clusterMaxBin_(ps.getParameter<std::vector<double>>("clusterMaxBin")) {}

L1TdeGEMTPG::~L1TdeGEMTPG() {}

void L1TdeGEMTPG::bookHistograms(DQMStore::IBooker& iBooker, const edm::Run&, const edm::EventSetup&) {
  iBooker.setCurrentFolder(monitorDir_);

  // chamber type
  for (unsigned iType = 0; iType < chambers_.size(); iType++) {
    // data vs emulator
    for (unsigned iData = 0; iData < dataEmul_.size(); iData++) {
      // cluster variable
      for (unsigned iVar = 0; iVar < clusterVars_.size(); iVar++) {
        const std::string key("cluster_" + clusterVars_[iVar] + "_" + dataEmul_[iData]);
        const std::string histName(key + "_" + chambers_[iType]);
        const std::string histTitle(chambers_[iType] + " Cluster " + clusterVars_[iVar] + " (" + dataEmul_[iData] +
                                    ")");
        chamberHistos[iType][key] =
            iBooker.book1D(histName, histTitle, clusterNBin_[iVar], clusterMinBin_[iVar], clusterMaxBin_[iVar]);
        chamberHistos[iType][key]->getTH1()->SetMinimum(0);
      }
    }
  }
}

void L1TdeGEMTPG::analyze(const edm::Event& e, const edm::EventSetup& c) {
  if (verbose_)
    edm::LogInfo("L1TdeGEMTPG") << "L1TdeGEMTPG: analyzing collections" << std::endl;

  // handles
  edm::Handle<GEMPadDigiClusterCollection> dataClusters;
  edm::Handle<GEMPadDigiClusterCollection> emulClusters;
  e.getByToken(data_token_, dataClusters);
  e.getByToken(emul_token_, emulClusters);

  for (auto it = dataClusters->begin(); it != dataClusters->end(); it++) {
    auto range = dataClusters->get((*it).first);
    const int type = ((*it).first).station() - 1;
    for (auto cluster = range.first; cluster != range.second; cluster++) {
      if (cluster->isValid()) {
        chamberHistos[type]["cluster_size_data"]->Fill(cluster->pads().size());
        chamberHistos[type]["cluster_pad_data"]->Fill(cluster->pads().front());
        chamberHistos[type]["cluster_bx_data"]->Fill(cluster->bx());
      }
    }
  }

  for (auto it = emulClusters->begin(); it != emulClusters->end(); it++) {
    auto range = emulClusters->get((*it).first);
    const int type = ((*it).first).station() - 1;
    for (auto cluster = range.first; cluster != range.second; cluster++) {
      if (cluster->isValid()) {
        chamberHistos[type]["cluster_size_emul"]->Fill(cluster->pads().size());
        chamberHistos[type]["cluster_pad_emul"]->Fill(cluster->pads().front());
        chamberHistos[type]["cluster_bx_emul"]->Fill(cluster->bx());
      }
    }
  }
}
