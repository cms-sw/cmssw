#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <iostream>
#include <stdlib.h>
#include <string>
#include <memory>
#include <vector>

class HLTTauDQMSummaryPlotter 
{
  
 public:

  HLTTauDQMSummaryPlotter(const edm::ParameterSet&);
  ~HLTTauDQMSummaryPlotter();
  void plot();


  
 private:

  void bookEfficiencyHisto(std::string,std::string,std::string,DQMStore*);
  void plotEfficiencyHisto(std::string,std::string,std::string ,std::string ,DQMStore*);

  void bookTriggerBitEfficiencyHistos(std::string folder,std::string histo,DQMStore*dbe);
  void plotTriggerBitEfficiencyHistos(std::string folder,std::string histo,DQMStore*dbe);

  std::vector<double> calcEfficiency(float,float);

  DQMStore *dbe;
  
  //Input Folders
  std::vector<std::string> L1Folder_;
  std::vector<std::string> caloFolder_;
  std::vector<std::string> trackFolder_;
  std::vector<std::string> pathFolder_;
  std::vector<std::string> litePathFolder_;

};
