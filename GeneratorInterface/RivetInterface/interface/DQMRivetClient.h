#ifndef GeneratorInterface_RivetInterface_DQMRivetClient_H
#define GeneratorInterface_RivetInterface_DQMRivetClient_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DQMServices/ClientConfig/interface/DQMGenericClient.h"
#include <set>
#include <string>
#include <vector>
#include <TH1.h>

class DQMStore;
class MonitorElement;

class DQMRivetClient : public edm::EDAnalyzer 
{
 public:
  DQMRivetClient(const edm::ParameterSet& pset);
  ~DQMRivetClient() {};

  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {};

  void endJob();

  /// EndRun
  void endRun(const edm::Run& r, const edm::EventSetup& c);
  
  struct LumiOption
  {
    std::string name, normHistName;
    double xsection;
  };

  struct ScaleFactorOption
  {
    std::string name;
    double scale;
  };

  void normalizeToIntegral(const std::string& startDir, const std::string& histName, const std::string& normHistName);
  void normalizeToLumi(const std::string& startDir, const std::string& histName, const std::string& normHistName, double xsection);
  void scaleByFactor(const std::string& startDir, const std::string& histName, double factor);

 private:
  unsigned int verbose_;

  DQMStore* theDQM;
  std::vector<std::string> subDirs_;
  std::string outputFileName_;

  std::vector<DQMGenericClient::NormOption> normOptions_;
  std::vector<LumiOption> lumiOptions_;
  std::vector<ScaleFactorOption> scaleOptions_;


};

#endif
