#ifndef __DQMOffline_PFTau_PFClient__
#define __DQMOffline_PFTau_PFClient__

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Utilities/interface/InputTag.h"


class DQMStore;
class MonitorElement;
class PFClient: public edm::EDAnalyzer {
 public:
  
  PFClient(const edm::ParameterSet& parameterSet);
  
 private:
  void beginJob();
  void analyze(edm::Event const&, edm::EventSetup const&){;}
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup);
  void endJob();

  void doSummaries();
  void createResolutionPlots(std::string& folder, std::string& name);
  void getHistogramParameters(MonitorElement* me_slice, double& avarage, double& rms, 
                                                        double& mean, double& sigma);
  std::vector<std::string> folderNames_;
  std::vector<std::string> histogramNames_;

  DQMStore* dqmStore_;

};

#endif 
