#ifndef __DQMOffline_PFTau_PFClient__
#define __DQMOffline_PFTau_PFClient__

#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"

class PFClient : public DQMEDHarvester {
public:
  PFClient(const edm::ParameterSet &parameterSet);

private:
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

  void doSummaries(DQMStore::IBooker &, DQMStore::IGetter &);
  void doEfficiency(DQMStore::IBooker &, DQMStore::IGetter &);
  void doProjection(DQMStore::IBooker &, DQMStore::IGetter &);
  void doProfiles(DQMStore::IBooker &, DQMStore::IGetter &);
  void createResolutionPlots(DQMStore::IBooker &, DQMStore::IGetter &, std::string &folder, std::string &name);
  void getHistogramParameters(MonitorElement *me_slice, double &avarage, double &rms, double &mean, double &sigma);
  void createEfficiencyPlots(DQMStore::IBooker &, DQMStore::IGetter &, std::string &folder, std::string &name);

  void createProjectionPlots(DQMStore::IBooker &, DQMStore::IGetter &, std::string &folder, std::string &name);
  void createProfilePlots(DQMStore::IBooker &, DQMStore::IGetter &, std::string &folder, std::string &name);

  std::vector<std::string> folderNames_;
  std::vector<std::string> histogramNames_;
  std::vector<std::string> effHistogramNames_;
  std::vector<std::string> projectionHistogramNames_;
  std::vector<std::string> profileHistogramNames_;
  bool efficiencyFlag_;
  bool profileFlag_;
};

#endif
