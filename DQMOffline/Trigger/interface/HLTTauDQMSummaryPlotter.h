// -*- c++ -*.
#ifndef HLTTauDQMSummaryPlotter_h
#define HLTTauDQMSummaryPlotter_h

#include "DQMOffline/Trigger/interface/HLTTauDQMPlotter.h"

class HLTTauDQMSummaryPlotter: private HLTTauDQMPlotter {
public:
    HLTTauDQMSummaryPlotter(const edm::ParameterSet& ps, const std::string& dqmBaseFolder, const std::string& type);
    HLTTauDQMSummaryPlotter(const std::string& dqmBaseFolder, const std::string& type);
    ~HLTTauDQMSummaryPlotter();

    using HLTTauDQMPlotter::isValid;

    void bookPlots();
    void plot();

private:
  const std::string type_;

  class SummaryPlotter;
  std::vector<std::unique_ptr<SummaryPlotter> > plotters_;
};
#endif
