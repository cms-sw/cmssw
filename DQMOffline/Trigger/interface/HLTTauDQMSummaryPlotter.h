// -*- c++ -*.
#ifndef HLTTauDQMSummaryPlotter_h
#define HLTTauDQMSummaryPlotter_h

#include "DQMOffline/Trigger/interface/HLTTauDQMPlotter.h"

class HLTTauDQMSummaryPlotter: private HLTTauDQMPlotter {
public:
    HLTTauDQMSummaryPlotter(const edm::ParameterSet& ps, const std::string& dqmBaseFolder);
    ~HLTTauDQMSummaryPlotter();

    using HLTTauDQMPlotter::isValid;

    void bookPlots();
    void plot();

private:
    void bookEfficiencyHisto(const std::string& folder, const std::string& name, const std::string& hist1, bool copyLabels=false);
    void plotEfficiencyHisto( std::string folder, std::string name, std::string hist1, std::string hist2 );
    void plotIntegratedEffHisto( std::string folder, std::string name, std::string refHisto, std::string evCount, int bin );
    void bookTriggerBitEfficiencyHistos( std::string folder, std::string histo );
    void plotTriggerBitEfficiencyHistos( std::string folder, std::string histo );
    void bookFractionHisto(const std::string& folder, const std::string& name);
    void plotFractionHisto(const std::string& folder, const std::string& name);

    std::string type_;
    DQMStore *store_;
};
#endif
