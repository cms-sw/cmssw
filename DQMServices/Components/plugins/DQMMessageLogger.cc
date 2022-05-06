// system includes
#include <cmath>
#include <vector>
#include <string>
#include <map>

// user includes
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/ErrorSummaryEntry.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/MessageLogger/interface/ELseverityLevel.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "TPad.h"

using namespace std;
using namespace edm;

class DQMMessageLogger : public DQMEDAnalyzer {
public:
  /// Constructor
  DQMMessageLogger(const edm::ParameterSet &);

  /// Destructor
  ~DQMMessageLogger() override = default;

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

DQMMessageLogger::DQMMessageLogger(const ParameterSet &parameters) {
  categories_errors = nullptr;
  categories_warnings = nullptr;
  modules_errors = nullptr;
  modules_warnings = nullptr;
  total_errors = nullptr;
  total_warnings = nullptr;

  //Get from cfg file
  categories_vector = parameters.getParameter<vector<string> >("Categories");
  directoryName = parameters.getParameter<string>("Directory");
  errorSummary_ = consumes<std::vector<edm::ErrorSummaryEntry> >(
      parameters.getUntrackedParameter<std::string>("errorSummary", "logErrorHarvester"));
}

void DQMMessageLogger::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const &iRun, edm::EventSetup const &iSetup) {
  metname = "errorAnalyzer";

  // MAKE CATEGORYMAP USING INPUT FROM CFG FILE
  for (unsigned int i = 0; i < categories_vector.size(); i++) {
    categoryMap.insert(pair<string, int>(categories_vector[i], i + 1));
  }

  // MAKE MODULEMAP
  using TNS = Service<edm::service::TriggerNamesService>;
  using stringvec = vector<std::string>;
  TNS tns;
  stringvec const &trigpaths = tns->getTrigPaths();

  for (auto const &trigpath : trigpaths) {
    stringvec strings = tns->getTrigPathModules(trigpath);

    for (auto &k : strings) {
      moduleMap.insert(pair<string, int>(k, moduleMap.size() + 1));
    }
  }

  // BOOK THE HISTOGRAMS
  LogTrace(metname) << "[DQMMessageLogger] Parameters initialization";

  if (!moduleMap.empty()) {
    ibooker.setCurrentFolder(directoryName + "/Errors");
    modules_errors = ibooker.book1D("modules_errors", "Errors per module", moduleMap.size(), 0, moduleMap.size());
    ibooker.setCurrentFolder(directoryName + "/Warnings");

    modules_warnings = ibooker.book1D("modules_warnings", "Warnings per module", moduleMap.size(), 0, moduleMap.size());

    for (auto it = moduleMap.begin(); it != moduleMap.end(); ++it) {
      modules_errors->setBinLabel((*it).second, (*it).first);
      modules_warnings->setBinLabel((*it).second, (*it).first);
    }
    modules_errors->getTH1()->GetXaxis()->LabelsOption("v");
    modules_warnings->getTH1()->GetXaxis()->LabelsOption("v");
  }

  if (!categoryMap.empty()) {
    ibooker.setCurrentFolder(directoryName + "/Errors");
    categories_errors =
        ibooker.book1D("categories_errors", "Errors per category", categoryMap.size(), 0, categoryMap.size());
    ibooker.setCurrentFolder(directoryName + "/Warnings");
    categories_warnings =
        ibooker.book1D("categories_warnings", "Warnings per category", categoryMap.size(), 0, categoryMap.size());

    for (auto it = categoryMap.begin(); it != categoryMap.end(); ++it) {
      categories_errors->setBinLabel((*it).second, (*it).first);
      categories_warnings->setBinLabel((*it).second, (*it).first);
    }
    categories_warnings->getTH1()->GetXaxis()->LabelsOption("v");
    categories_errors->getTH1()->GetXaxis()->LabelsOption("v");
  }

  // HOW MANY BINS SHOULD THE ERROR HIST HAVE?
  int nbins = 11;
  total_warnings = ibooker.book1D("total_warnings", "Total warnings per event", nbins, -0.5, nbins + 0.5);
  ibooker.setCurrentFolder(directoryName + "/Errors");
  total_errors = ibooker.book1D("total_errors", "Total errors per event", nbins, -0.5, nbins + 0.5);

  for (int i = 0; i < nbins; ++i) {
    stringstream out;
    out << i;
    string s = out.str();
    total_errors->setBinLabel(i + 1, s);
    total_warnings->setBinLabel(i + 1, s);
  }
}

void DQMMessageLogger::analyze(const Event &iEvent, const EventSetup &iSetup) {
  LogTrace(metname) << "[DQMMessageLogger] Analysis of event # ";

  // Take the ErrorSummaryEntry container
  Handle<std::vector<edm::ErrorSummaryEntry> > errors;
  iEvent.getByToken(errorSummary_, errors);
  // Check that errors is valid
  if (!errors.isValid()) {
    return;
  }
  // Compare severity level of error with ELseveritylevel instance el : "-e" should be the lowest error
  ELseverityLevel el(ELseverityLevel::ELsev_error);

  // Find the total number of errors in iEvent
  if (errors->empty()) {
    if (total_errors != nullptr) {
      total_errors->Fill(0);
    }
    if (total_warnings != nullptr) {
      total_warnings->Fill(0);
    }
  } else {
    int e = 0;
    int w = 0;
    for (int i = 0, n = errors->size(); i < n; i++) {
      if ((*errors)[i].severity.getLevel() < el.getLevel()) {
        w += (*errors)[i].count;
      } else {
        e += (*errors)[i].count;
      }
    }
    if (total_errors != nullptr) {
      total_errors->Fill(e);
    }
    if (total_warnings != nullptr) {
      total_warnings->Fill(w);
    }
  }

  for (int i = 0, n = errors->size(); i < n; i++) {
    //cout << "Severity for error/warning: " << (*errors)[i].severity << " " <<(*errors)[i].module  << endl;

    if (!errors->empty()) {
      // IF THIS IS AN ERROR on the ELseverityLevel SCALE, FILL ERROR HISTS
      if ((*errors)[i].severity.getLevel() >= el.getLevel()) {
        if (categories_errors != nullptr) {
          auto it = categoryMap.find((*errors)[i].category);
          if (it != categoryMap.end()) {
            // FILL THE RIGHT BIN
            categories_errors->Fill((*it).second - 1, (*errors)[i].count);
          }
        }
        //	if (categoryECount.size()<=40)
        //	  categoryECount[(*errors)[i].category]+=(*errors)[i].count;

        if (modules_errors != nullptr) {
          // remove the first part of the module string, what is before ":"
          string s = (*errors)[i].module;
          size_t pos = s.find(':');
          string s_temp = s.substr(pos + 1, s.size());
          auto it = moduleMap.find(s_temp);
          if (it != moduleMap.end()) {
            // FILL THE RIGHT BIN
            modules_errors->Fill((*it).second - 1, (*errors)[i].count);
          }
        }
        // IF ONLY WARNING, FILL WARNING HISTS
      } else {
        if (categories_warnings != nullptr) {
          auto it = categoryMap.find((*errors)[i].category);
          if (it != categoryMap.end()) {
            // FILL THE RIGHT BIN
            categories_warnings->Fill((*it).second - 1, (*errors)[i].count);
          }
        }

        //	if (categoryWCount.size()<=40)
        //	  categoryWCount[(*errors)[i].category]+=(*errors)[i].count;

        if (modules_warnings != nullptr) {
          // remove the first part of the module string, what is before ":"
          string s = (*errors)[i].module;
          size_t pos = s.find(':');
          string s_temp = s.substr(pos + 1, s.size());
          auto it = moduleMap.find(s_temp);
          if (it != moduleMap.end()) {
            // FILL THE RIGHT BIN
            modules_warnings->Fill((*it).second - 1, (*errors)[i].count);
          }
        }
      }
    }
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DQMMessageLogger);
