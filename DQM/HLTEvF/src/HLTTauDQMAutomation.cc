#include "DQM/HLTEvF/interface/HLTTauDQMAutomation.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <boost/regex.hpp>

HLTTauDQMAutomation::HLTTauDQMAutomation( ) {
}

HLTTauDQMAutomation::HLTTauDQMAutomation( std::string hltProcessName, double L1MatchDr, double HLTMatchDr ) {
    hltProcessName_ = hltProcessName;
    L1MatchDr_ = L1MatchDr;
    HLTMatchDr_ = HLTMatchDr;
}

HLTTauDQMAutomation::~HLTTauDQMAutomation() {
}

void HLTTauDQMAutomation::AutoCompleteConfig( std::vector<edm::ParameterSet>& config, HLTConfigProvider const& HLTCP ) {
    //Find tau trigger paths
    boost::regex reTau(".*Tau.*");
    std::map<std::string,HLTTauDQMFilter> filters;
    if ( HLTCP.inited() ) {
        for ( std::vector<std::string>::const_iterator ipath = HLTCP.triggerNames().begin(); ipath != HLTCP.triggerNames().end(); ++ipath ) {
            if ( boost::regex_match(*ipath, reTau) && HLTCP.prescaleValue(0,*ipath) > 0 ) {
                HLTTauDQMFilter tmp(*ipath, HLTCP.prescaleValue(0,*ipath), hltProcessName_, L1MatchDr_, HLTMatchDr_);
                selectHLTTauDQMFilter(filters, tmp);
            }
        }
    }
    
    //Add PFTau paths
    boost::regex reMuTau("Mu(.+?)Tau");
    boost::smatch what;
    std::map<std::string,HLTTauDQMFilter> pfTauFilters;
    for ( std::map<std::string,HLTTauDQMFilter>::const_iterator ipath = filters.begin(); ipath != filters.end(); ++ipath ) {
        std::string::const_iterator start = ipath->first.begin();
        std::string::const_iterator end = ipath->first.end();
        if ( boost::regex_match(start, end, what, reMuTau) ) {
            HLTTauDQMFilter tmp(ipath->second.name(), what[1]+"PFTau", HLTCP.prescaleValue(0,ipath->second.name()), hltProcessName_, L1MatchDr_, HLTMatchDr_, 1, 0, 0);
            selectHLTTauDQMFilter(pfTauFilters, tmp);
        }
    }
    filters.insert(pfTauFilters.begin(),pfTauFilters.end());
    
    //Auto configuration
    std::vector<edm::ParameterSet>::iterator lpsum = config.end();
    std::vector<std::map<std::string,HLTTauDQMFilter>::iterator> selectedFilters;
    for ( std::vector<edm::ParameterSet>::iterator iconfig = config.begin(); iconfig != config.end(); ++iconfig ) {
        std::string configtype;
        try {
            configtype = iconfig->getUntrackedParameter<std::string>("ConfigType");
        } catch ( cms::Exception &e ) {
            edm::LogWarning("HLTTauDQMAutomation") << e.what() << std::endl;
            continue;
        }
        if (configtype == "Path") {
            try {
                if ( iconfig->getUntrackedParameter<std::vector<edm::ParameterSet> >("Filters",std::vector<edm::ParameterSet>()).size() == 0 ) {
                    std::string triggerName = iconfig->getUntrackedParameter<std::string>("DQMFolder");
                    std::map<std::string,HLTTauDQMFilter>::iterator iter = findFilter(filters,triggerName);
                    if ( iter != filters.end() ) {
                        iconfig->addUntrackedParameter<std::string>("DQMFolder", iter->first);
                        iconfig->addUntrackedParameter<std::vector<edm::ParameterSet> >("Filters", iter->second.getPSets(HLTCP) );
                        iconfig->addUntrackedParameter<edm::ParameterSet >("Reference", iter->second.getRefPSet());
                        selectedFilters.push_back(iter);
                    }
                }
            } catch ( cms::Exception &e ) {
                edm::LogWarning("HLTTauDQMAutomation") << e.what() << std::endl;
                continue;
            }
        }
        else if (configtype == "LitePath") {
            try {
                if ( iconfig->getUntrackedParameter<std::vector<edm::ParameterSet> >("Filters",std::vector<edm::ParameterSet>()).size() == 0 ) {
                    std::string triggerName = iconfig->getUntrackedParameter<std::string>("DQMFolder");
                    if ( triggerName != "Summary" ) {
                        std::map<std::string,HLTTauDQMFilter>::iterator iter = findFilter(filters,triggerName);
                        if ( iter != filters.end() ) {
                            iconfig->addUntrackedParameter<std::string>("DQMFolder", iter->first);
                            iconfig->addUntrackedParameter<std::vector<edm::ParameterSet> >("Filters", iter->second.getPSets(HLTCP) );
                            selectedFilters.push_back(iter);
                        }
                    } else {
                        lpsum = iconfig;
                    }
                }
            } catch ( cms::Exception &e ) {
                edm::LogWarning("HLTTauDQMAutomation") << e.what() << std::endl;
                continue;
            }
        }
    }
    
    //Configure litepath summary
    if ( lpsum != config.end() ) {
        std::vector<edm::ParameterSet> filterSet;
        for ( std::vector<std::map<std::string,HLTTauDQMFilter>::iterator>::iterator ifilter = selectedFilters.begin(); ifilter != selectedFilters.end(); ++ifilter ) {
            if ( !((*ifilter)->second.getSummaryPSet(HLTCP).empty()) ) {
                filterSet.push_back((*ifilter)->second.getSummaryPSet(HLTCP));
            }
        }
        lpsum->addUntrackedParameter<std::vector<edm::ParameterSet> >("Filters", filterSet );
    }
}

void HLTTauDQMAutomation::AutoCompleteMatching( edm::ParameterSet& config, HLTConfigProvider const& HLTCP, std::string moduleType ) {
    //Find tau trigger paths
    boost::regex reTau(".*Tau.*");
    std::map<std::string,HLTTauDQMFilter> filters;
    if ( HLTCP.inited() ) {
        for ( std::vector<std::string>::const_iterator ipath = HLTCP.triggerNames().begin(); ipath != HLTCP.triggerNames().end(); ++ipath ) {
            if ( boost::regex_match(*ipath, reTau) && HLTCP.prescaleValue(0,*ipath) > 0 ) {
                HLTTauDQMFilter tmp(*ipath, HLTCP.prescaleValue(0,*ipath), hltProcessName_, L1MatchDr_, HLTMatchDr_);
                selectHLTTauDQMFilter(filters, tmp);
            }
        }
    }
    
    //Auto configuration
    if ( config.getUntrackedParameter<bool>("doMatching") ) {
        std::vector<edm::ParameterSet> matchingFilters = config.getUntrackedParameter<std::vector<edm::ParameterSet> >("matchFilters");
        for ( std::vector<edm::ParameterSet>::iterator imatch = matchingFilters.begin(); imatch != matchingFilters.end(); ++imatch ) {
            std::string autoFilterName = imatch->getUntrackedParameter<std::string>("AutomaticFilterName","");
            if ( autoFilterName != "" ) {
                try {
                    std::map<std::string,HLTTauDQMFilter>::iterator iter = findFilter(filters,autoFilterName);
                    if ( iter != filters.end() ) {
                        std::map<int,std::string> modules = iter->second.interestingModules(HLTCP);
                        boost::regex exprEle(moduleType);
                        std::string selectedModule = "";
                        for ( std::map<int,std::string>::const_iterator imodule = modules.begin(); imodule != modules.end(); ++imodule ) {
                            std::string::const_iterator start = HLTCP.moduleType(imodule->second).begin();
                            std::string::const_iterator end = HLTCP.moduleType(imodule->second).end();
                            if ( boost::regex_match(start, end, exprEle) ) {
                                selectedModule = imodule->second;
                                break;
                            }
                        }
                        
                        imatch->addUntrackedParameter<edm::InputTag>("FilterName", edm::InputTag(selectedModule,"",hltProcessName_) );
                    } else {
                        imatch->addUntrackedParameter<edm::InputTag>("FilterName", edm::InputTag("","",hltProcessName_) );
                    }
                } catch ( cms::Exception &e ) {
                    edm::LogWarning("HLTTauDQMAutomation") << e.what() << std::endl;
                    continue;
                }
            }
        }
        config.addUntrackedParameter<std::vector<edm::ParameterSet> >("matchFilters", matchingFilters );
    }
}

bool HLTTauDQMAutomation::selectHLTTauDQMFilter(std::map<std::string,HLTTauDQMFilter>& container, HLTTauDQMFilter const& filter) {
    bool inserted = false;
    std::map<std::string,HLTTauDQMFilter>::iterator iter = container.find(filter.type());
    if ( iter == container.end() ) {
        container.insert(std::make_pair(filter.type(), filter));
        inserted = true;
    } else {
        iter->second = iter->second.select(filter,inserted);
    }
    return inserted;
}

std::map<std::string,HLTTauDQMFilter>::iterator HLTTauDQMAutomation::findFilter(std::map<std::string,HLTTauDQMFilter>& container, std::string const& triggerName) {
    boost::regex re("[a-zA-Z_]+");
    std::map<std::string,HLTTauDQMFilter>::iterator iter;
    if ( boost::regex_match(triggerName, re) ) {
        iter = container.find(triggerName);
        return iter;
    } else {
        boost::regex tmpRe(triggerName);
        for ( iter = container.begin(); iter != container.end(); ++iter ) {
            if ( boost::regex_match(iter->first, tmpRe) ) { //Always take the first matching path
                return iter;
            }
        }
    }
    return container.end();
}
