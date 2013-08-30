#ifndef HLTTauDQMSummaryPlotter_h
#define HLTTauDQMSummaryPlotter_h

#include "DQMOffline/Trigger/interface/HLTTauDQMPlotter.h"

class HLTTauDQMSummaryPlotter : public HLTTauDQMPlotter {
public:
    HLTTauDQMSummaryPlotter(const edm::ParameterSet& ps, const std::string& dqmBaseFolder);
    ~HLTTauDQMSummaryPlotter();
    const std::string name() { return name_; }

    void bookPlots();
    void plot();
    
private:
    void bookEfficiencyHisto( std::string folder, std::string name, std::string hist1 );
    void plotEfficiencyHisto( std::string folder, std::string name, std::string hist1, std::string hist2 );
    void plotIntegratedEffHisto( std::string folder, std::string name, std::string refHisto, std::string evCount, int bin );
    void bookTriggerBitEfficiencyHistos( std::string folder, std::string histo );
    void plotTriggerBitEfficiencyHistos( std::string folder, std::string histo );
    void bookFractionHisto(const std::string& folder, const std::string& name);
    void plotFractionHisto(const std::string& folder, const std::string& name);
    std::pair<double,double> calcEfficiency( float num, float denom );
    
    std::string type_;
};
#endif
