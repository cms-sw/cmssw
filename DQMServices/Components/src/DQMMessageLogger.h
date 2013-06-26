
#ifndef DQMMESSAGELOGGER_H
#define DQMMESSAGELOGGER_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/FWLite/interface/Event.h"
#include<vector>
#include <string>
#include <map>

class DQMStore;
class MonitorElement;

class DQMMessageLogger : public edm::EDAnalyzer {
 public:

  /// Constructor
  DQMMessageLogger(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~DQMMessageLogger();
  
  /// Inizialize parameters for histo binning
  void beginJob();

  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&);


  /// collate categories in summary plots
  void endRun(const edm::Run & r, const edm::EventSetup & c);

  /// Save the histos
  void endJob();

 private:


  // ----------member data ---------------------------
  
  DQMStore* theDbe;
  // Switch for verbosity
  std::string metname;
  
  std::map<std::string,int> moduleMap;
  std::map<std::string,int> categoryMap;
  std::map<std::string,int> categoryWCount;
  std::map<std::string,int> categoryECount;
  // from parameters
  std::vector<std::string> categories_vector;
  std::string directoryName;

  //The histos
  MonitorElement *categories_errors;
  MonitorElement *categories_warnings;
  MonitorElement *modules_errors;
  MonitorElement *modules_warnings;
  MonitorElement *total_errors;
  MonitorElement *total_warnings;

  
};
#endif  


