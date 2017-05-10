#ifndef __DQMOffline_PFTau_PFClient__
#define __DQMOffline_PFTau_PFClient__

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMEDHarvester.h"

class DQMStore;
class MonitorElement;
class PFClient: public DQMEDHarvester {
 public:
  
  PFClient(const edm::ParameterSet& parameterSet);
  
 private:
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup);
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override; 

  void doSummaries();
  void doEfficiency();
  void doProjection();
  void doProfiles();
  void createResolutionPlots(std::string& folder, std::string& name);
  void getHistogramParameters(MonitorElement* me_slice, double& avarage, double& rms, 
                                                        double& mean, double& sigma);
  void createEfficiencyPlots(std::string& folder, std::string& name);

  void createProjectionPlots(std::string& folder, std::string& name);
  void createProfilePlots(std::string& folder, std::string& name);
     
  std::vector<std::string> folderNames_;
  std::vector<std::string> histogramNames_;
  std::vector<std::string> effHistogramNames_;
  std::vector<std::string> projectionHistogramNames_;
  std::vector<std::string> profileHistogramNames_;
  bool efficiencyFlag_;
  bool profileFlag_;

  DQMStore::IBooker * ibooker_;
  DQMStore::IGetter * igetter_;

};

#endif 
