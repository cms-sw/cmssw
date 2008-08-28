#ifndef DQMOffline_Trigger_HLTTauPostProcessor_h
#define DQMOffline_Trigger_HLTTauPostProcessor_h


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

class HLTTauPostProcessor : public edm::EDAnalyzer
{
  
 public:

  explicit HLTTauPostProcessor(const edm::ParameterSet&);
  virtual ~HLTTauPostProcessor();
  virtual void beginJob(const edm::EventSetup&);
  virtual void endJob();  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  virtual void endRun(const edm::Run&, const edm::EventSetup&);


  
 private:
  void createEfficiencyHisto(std::string,std::string,std::string ,std::string ,DQMStore*);
  void createIntegratedHisto(std::string ,std::string ,std::string ,int ,DQMStore* );
  void calculatePathEfficiencies(std::string folder,std::string histo,DQMStore*dbe);
  std::vector<double> calcEfficiency(float,float);

  DQMStore *dbe;
  
  //Input Folders
  std::vector<std::string> L1Folder_;
  std::vector<std::string> L2Folder_;
  std::vector<std::string> L25Folder_;
  std::vector<std::string> L3Folder_;

  //Path Validation Folder
  std::vector<std::string> pathValFolder_;

  //Offline DQM Path Comparison Folder
  std::vector<std::string> pathDQMFolder_;

};

#endif
