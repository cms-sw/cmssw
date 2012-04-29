#ifndef SusyPostProcessor_H
#define SusyPostProcessor_H

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//

#include <vector>
#include <string>

#include "TFile.h"
#include "TH1.h"
#include "TMath.h"

class SusyPostProcessor : public edm::EDAnalyzer
{
 public:
  explicit SusyPostProcessor( const edm::ParameterSet& pSet ) ;
  ~SusyPostProcessor();
                                   
      
  virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup ) ;
  virtual void beginJob(void) ;
  virtual void beginRun(const edm::Run&, const edm::EventSetup& iSetup);
  virtual void endJob();
  void endRun(const edm::Run& , const edm::EventSetup& ) ;

 private:

  edm::ParameterSet iConfig;
  void QuantilePlots(MonitorElement* ME, double q_value);

  std::string SUSYFolder;

  DQMStore* dqm;
  std::vector<MonitorElement*> histoVector;
  std::vector<std::string> Dirs;
  
};

#endif
