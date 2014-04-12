
#ifndef DQMMESSAGELOGGERCLIENT_H
#define DQMMESSAGELOGGERCLIENT_H


#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <vector>
#include <string>
#include <map>

class DQMMessageLoggerClient : public edm::EDAnalyzer {
 public:
  // Constructor
  DQMMessageLoggerClient(const edm::ParameterSet&);
  // Destructor
  virtual ~DQMMessageLoggerClient();
  
 protected:
  
  void beginJob();
  //void beginRun(const edm::Run&, const edm::EventSetup&);

  void beginRun(const edm::Run&, const edm::EventSetup&);
  
  
  // Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&);
  void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);

  void endLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup&);
  
  // Save the histos
  void endRun(const edm::Run&, const edm::EventSetup&);
  void endJob();

 private:

  void fillHistograms();

  // ----------member data ---------------------------
  
  DQMStore* theDbe;
  edm::ParameterSet parameters;
  std::string directoryName;

  std::vector<std::string> binLabel;
  std::vector<Double_t> binContent;

  int nBinsErrors;
  int nBinsWarnings;

  MonitorElement *modulesErrorsFound;
  MonitorElement *modulesWarningsFound;
  MonitorElement *categoriesErrorsFound;
  MonitorElement *categoriesWarningsFound;

};
#endif  


