#ifndef HLTTauDQMAutomation_h
#define HLTTauDQMAutomation_h

#include <iostream>
#include <map>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DQM/HLTEvF/interface/HLTTauDQMFilter.h"

class HLTTauDQMAutomation {
public:
    HLTTauDQMAutomation();
    HLTTauDQMAutomation( std::string hltProcessName, double L1MatchDr, double HLTMatchDr );
    virtual ~HLTTauDQMAutomation();
    void AutoCompleteConfig( std::vector<edm::ParameterSet>& config, HLTConfigProvider const& HLTCP );
    void AutoCompleteMatching( edm::ParameterSet& config, HLTConfigProvider const& HLTCP, std::string moduleType );
    
private:
    std::string hltProcessName_;
    double L1MatchDr_;
    double HLTMatchDr_;
    
    //Helper functions
    bool selectHLTTauDQMFilter(std::map<std::string,HLTTauDQMFilter>& container, HLTTauDQMFilter const& filter);
    
    std::map<std::string,HLTTauDQMFilter>::iterator findFilter(std::map<std::string,HLTTauDQMFilter>& container, std::string const& triggerName);
};

#endif
