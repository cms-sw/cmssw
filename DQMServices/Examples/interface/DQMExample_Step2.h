#ifndef DQMExample_Step2_H
#define DQMExample_Step2_H

//Framework
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//DQM
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
 
class DQMExample_Step2: public edm::EDAnalyzer{

public:

  DQMExample_Step2(const edm::ParameterSet& ps);
  virtual ~DQMExample_Step2();
  
protected:

  void beginJob();
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup);
  void analyze(edm::Event const& e, edm::EventSetup const& eSetup);
  void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& eSetup) ;
  void endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& eSetup);
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup);
  void endJob();

private:

  //private variables
  DQMStore* dbe_;  

  //variables from config file
  std::string numMonitorName_;
  std::string denMonitorName_;

  // Histograms
  MonitorElement* h_ptRatio;

};


#endif
