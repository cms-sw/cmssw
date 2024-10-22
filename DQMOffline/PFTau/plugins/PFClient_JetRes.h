#ifndef __DQMOffline_PFTau_PFClient_JetRes__
#define __DQMOffline_PFTau_PFClient_JetRes__

#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"

class PFClient_JetRes : public DQMEDHarvester {
public:
  PFClient_JetRes(const edm::ParameterSet &parameterSet);

private:
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

  void doSummaries(DQMStore::IBooker &, DQMStore::IGetter &);
  void doEfficiency(DQMStore::IBooker &, DQMStore::IGetter &);
  void createResolutionPlots(DQMStore::IBooker &, DQMStore::IGetter &, std::string &folder, std::string &name);
  void getHistogramParameters(MonitorElement *me_slice, double &avarage, double &rms, double &mean, double &sigma);
  void createEfficiencyPlots(DQMStore::IBooker &, DQMStore::IGetter &, std::string &folder, std::string &name);

  std::vector<std::string> folderNames_;
  std::vector<std::string> histogramNames_;
  std::vector<std::string> effHistogramNames_;
  std::vector<int> PtBins_;

  bool efficiencyFlag_;
};

#endif
