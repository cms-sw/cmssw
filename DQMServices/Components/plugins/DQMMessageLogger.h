
#ifndef DQMMESSAGELOGGER_H
#define DQMMESSAGELOGGER_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/Common/interface/ErrorSummaryEntry.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <vector>
#include <string>
#include <map>

class DQMMessageLogger : public DQMEDAnalyzer {
public:
  /// Constructor
  DQMMessageLogger(const edm::ParameterSet &);

  /// Destructor
  ~DQMMessageLogger() override;

  /// Get the analysis
  void analyze(const edm::Event &, const edm::EventSetup &) override;

protected:
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

private:
  // ----------member data ---------------------------

  // Switch for verbosity
  std::string metname;

  std::map<std::string, int> moduleMap;
  std::map<std::string, int> categoryMap;
  std::map<std::string, int> categoryWCount;
  std::map<std::string, int> categoryECount;
  // from parameters
  std::vector<std::string> categories_vector;
  std::string directoryName;
  edm::EDGetTokenT<std::vector<edm::ErrorSummaryEntry> > errorSummary_;

  //The histos
  MonitorElement *categories_errors;
  MonitorElement *categories_warnings;
  MonitorElement *modules_errors;
  MonitorElement *modules_warnings;
  MonitorElement *total_errors;
  MonitorElement *total_warnings;
};
#endif
