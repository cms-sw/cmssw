#ifndef CALOTOWERANALYZER_H
#define CALOTOWERANALYZER_H

// author: Bobby Scurlock (The University of Florida)
// date: 8/24/2006
// modification: Mike Schmitt
// date: 02.28.2007
// note: code rewrite

#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/METReco/interface/HcalNoiseSummary.h"

#include <string>
#include <map>
#include "DQMServices/Core/interface/MonitorElement.h"

class CaloTowerAnalyzer: public edm::EDAnalyzer {
public:

  explicit CaloTowerAnalyzer(const edm::ParameterSet&);

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginRun(const edm::Run& ,const edm::EventSetup&);
  //virtual void beginJob();
  virtual void endJob();

private:

  // DAQ Tools
  DQMStore* dbe_;

  // Inputs from Configuration
  edm::InputTag caloTowersLabel_;
  std::vector< edm::InputTag >  HLTBitLabel_ ;
  edm::InputTag HLTResultsLabel_;
  edm::InputTag HcalNoiseSummaryTag_;
  bool debug_;
  double energyThreshold_;
  bool allhist_;
  bool finebinning_;
  bool hltselection_;
  std::string FolderName_;
  int Nevents;


  MonitorElement* hCT_Nevents;
  MonitorElement* hCT_et_ieta_iphi;
  MonitorElement* hCT_emEt_ieta_iphi;
  MonitorElement* hCT_hadEt_ieta_iphi;
  MonitorElement* hCT_outerEt_ieta_iphi;
  MonitorElement* hCT_Occ_ieta_iphi;
  MonitorElement* hCT_Occ_EM_Et_ieta_iphi;
  MonitorElement* hCT_Occ_HAD_Et_ieta_iphi;
  MonitorElement* hCT_Occ_Outer_Et_ieta_iphi;
  MonitorElement* hCT_etvsieta;
  MonitorElement* hCT_Minetvsieta;
  MonitorElement* hCT_Maxetvsieta;
  MonitorElement* hCT_emEtvsieta;
  MonitorElement* hCT_hadEtvsieta;
  MonitorElement* hCT_outerEtvsieta;
  MonitorElement* hCT_Occvsieta;
  MonitorElement* hCT_SETvsieta;
  MonitorElement* hCT_METvsieta;
  MonitorElement* hCT_METPhivsieta;
  MonitorElement* hCT_MExvsieta;
  MonitorElement* hCT_MEyvsieta;
  std::vector<MonitorElement*> hCT_NEvents_HLT;
};

#endif
