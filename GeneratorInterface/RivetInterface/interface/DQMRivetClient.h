#ifndef GeneratorInterface_RivetInterface_DQMRivetClient_H
#define GeneratorInterface_RivetInterface_DQMRivetClient_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <set>
#include <string>
#include <vector>
#include <TH1.h>

class DQMRivetClient : public edm::EDAnalyzer {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;
  struct NormOption {
    std::string name, normHistName;
  };

  DQMRivetClient(const edm::ParameterSet& pset);
  ~DQMRivetClient() override{};

  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) override{};

  void endJob() override;

  /// EndRun
  void endRun(const edm::Run& r, const edm::EventSetup& c) override;

  struct LumiOption {
    std::string name, normHistName;
    double xsection;
  };

  struct ScaleFactorOption {
    std::string name;
    double scale;
  };

  void normalizeToIntegral(const std::string& startDir, const std::string& histName, const std::string& normHistName);
  void normalizeToLumi(const std::string& startDir,
                       const std::string& histName,
                       const std::string& normHistName,
                       double xsection);
  void scaleByFactor(const std::string& startDir, const std::string& histName, double factor);

private:
  unsigned int verbose_;

  DQMStore* theDQM;
  std::vector<std::string> subDirs_;
  std::string outputFileName_;

  std::vector<NormOption> normOptions_;
  std::vector<LumiOption> lumiOptions_;
  std::vector<ScaleFactorOption> scaleOptions_;
};

#endif
