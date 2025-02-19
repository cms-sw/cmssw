#include "DQMOffline/Trigger/interface/HLTTauDQMFilter.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include <boost/regex.hpp>

HLTTauDQMFilter::HLTTauDQMFilter( std::string const& name, int initialPrescale, std::string hltTauDQMProcess, double L1MatchDeltaR, double HLTMatchDeltaR ) {
    name_ = name;
    alias_ = "";
    initialPrescale_ = initialPrescale;
    hltTauDQMProcess_ = hltTauDQMProcess;
    L1MatchDeltaR_ = L1MatchDeltaR;
    HLTMatchDeltaR_ = HLTMatchDeltaR;
    count_taus_ = 0;
    count_muons_ = 0;
    count_electrons_ = 0;
    count_mets_ = 0;
    count_quadjets_ = 0;
    count_pfmhts_ = 0;
    count_hts_ = 0;
    regexSearch();
    setType();
}

HLTTauDQMFilter::HLTTauDQMFilter( std::string const& name, std::string const& alias, int initialPrescale, std::string hltTauDQMProcess, double L1MatchDeltaR, double HLTMatchDeltaR, int nRefTaus, int nRefElectrons, int nRefMuons ) {
    name_ = name;
    alias_ = alias;
    initialPrescale_ = initialPrescale;
    hltTauDQMProcess_ = hltTauDQMProcess;
    L1MatchDeltaR_ = L1MatchDeltaR;
    HLTMatchDeltaR_ = HLTMatchDeltaR;
    count_taus_ = 0;
    count_muons_ = 0;
    count_electrons_ = 0;
    count_mets_ = 0;
    count_quadjets_ = 0;
    count_pfmhts_ = 0;
    count_hts_ = 0;
    regexSearch();
    setType();
    if (nRefTaus != -1) count_taus_ = nRefTaus;
    if (nRefElectrons != -1) count_electrons_ = nRefElectrons;
    if (nRefMuons != -1) count_muons_ = nRefMuons;
}

HLTTauDQMFilter::~HLTTauDQMFilter() {
}

void HLTTauDQMFilter::print() {
    std::cout << "HLTTauDQMFilter '" << name_ << "':" << std::endl;
    std::cout << " initial prescale: " << initialPrescale_ << std::endl;
    std::cout << " " << count_taus_ << " tau(s)" << std::endl;
    std::cout << " " << count_muons_ << " muon(s)" << std::endl;
    std::cout << " " << count_electrons_ << " electron(s)" << std::endl;
    std::cout << " " << count_mets_ << " MET(s)" << std::endl;
    std::cout << " " << count_quadjets_ << " QuadJet(s)" << std::endl;
    std::cout << " " << count_pfmhts_ << " PFMHT(s)" << std::endl;
    std::cout << " " << count_hts_ << " HT(s)" << std::endl;
    std::cout << " --> type: " << type() << std::endl;
}

int HLTTauDQMFilter::leadingTau() const {
    if ( taus_.size() > 0 ) {
        std::multimap<int,std::string>::const_iterator iter = taus_.end(); --iter;
        return iter->first;
    }
    return 0;
}

int HLTTauDQMFilter::leadingMuon() const {
    if ( muons_.size() > 0 ) {
        std::multimap<int,std::string>::const_iterator iter = muons_.end(); --iter;
        return iter->first;
    }
    return 0;
}

int HLTTauDQMFilter::leadingElectron() const {
    if ( electrons_.size() > 0 ) {
        std::multimap<int,std::string>::const_iterator iter = electrons_.end(); --iter;
        return iter->first;
    }
    return 0;
}

int HLTTauDQMFilter::leadingMET() const {
    if ( mets_.size() > 0 ) {
        std::multimap<int,std::string>::const_iterator iter = mets_.end(); --iter;
        return iter->first;
    }
    return 0;
}

int HLTTauDQMFilter::leadingQuadJet() const {
    if ( quadjets_.size() > 0 ) {
        std::multimap<int,std::string>::const_iterator iter = quadjets_.end(); --iter;
        return iter->first;
    }
    return 0;
}

int HLTTauDQMFilter::leadingPFMHT() const {
    if ( pfmht_.size() > 0 ) {
        std::multimap<int,std::string>::const_iterator iter = pfmht_.end(); --iter;
        return iter->first;
    }
    return 0;
}

int HLTTauDQMFilter::leadingHT() const {
    if ( ht_.size() > 0 ) {
        std::multimap<int,std::string>::const_iterator iter = ht_.end(); --iter;
        return iter->first;
    }
    return 0;
}

HLTTauDQMFilter const& HLTTauDQMFilter::select( HLTTauDQMFilter const& filter, bool& swapped ) {
    if ( filter.initialPrescale() > 0 ) {
        if ( filter.initialPrescale() <= initialPrescale() ) {
            if ( leadingTau() > filter.leadingTau() ) {
                swapped = true;
                return filter;
            } else if ( leadingTau() == filter.leadingTau() ) {
                if ( leadingMuon() > filter.leadingMuon() || leadingElectron() > filter.leadingElectron() || leadingMET() > filter.leadingMET() || leadingQuadJet() > filter.leadingQuadJet() || leadingPFMHT() > filter.leadingPFMHT() || leadingHT() > filter.leadingHT() ) {
                    swapped = true;
                    return filter;
                }
            }
        }
    }
    swapped = false;
    return *this;
}

void HLTTauDQMFilter::regexSearch() {
    boost::regex exprTau("([a-zA-Z]*?)IsoPFTau([0-9]+)_"); 
    boost::smatch what;
    std::string::const_iterator start = name_.begin();
    std::string::const_iterator end = name_.end();
    while ( boost::regex_search(start, end, what, exprTau) ) {
        int energy = 0;
        string2int(what[2].str().c_str(),energy);
        if (std::string(what[1]) == "Double" || std::string(what[1]) == "DoubleLoose" || std::string(what[1]) == "DoubleMedium" || std::string(what[1]) == "DoubleTight") {
            taus_.insert(std::pair<int,std::string>(energy,""));
            taus_.insert(std::pair<int,std::string>(energy,""));
            count_taus_ += 2;
        } else {
            taus_.insert(std::pair<int,std::string>(energy,what[1]));
            count_taus_++;
        }
        
        // update search position: 
        start = what[0].second;
    }
    
    boost::regex exprMuon("([a-zA-Z]*?)Mu([0-9]+)"); 
    start = name_.begin();
    while ( boost::regex_search(start, end, what, exprMuon) ) {
        int energy = 0;
        string2int(what[2].str().c_str(),energy);
        muons_.insert(std::pair<int,std::string>(energy,what[1]));
        count_muons_++;
        
        // update search position: 
        start = what[0].second;
    }
    
    boost::regex exprElectron("([a-zA-Z]*?)Ele([0-9]+)"); 
    start = name_.begin();    
    while ( boost::regex_search(start, end, what, exprElectron) ) {
        int energy = 0;
        string2int(what[2].str().c_str(),energy);
        electrons_.insert(std::pair<int,std::string>(energy,what[1]));
        count_electrons_++;
        
        // update search position: 
        start = what[0].second;
    }
    
    boost::regex exprMET("([a-zA-Z]*?)MET([0-9]+)"); 
    start = name_.begin();
    while ( boost::regex_search(start, end, what, exprMET) ) {
        int energy = 0;
        string2int(what[2].str().c_str(),energy);
        mets_.insert(std::pair<int,std::string>(energy,what[1]));
        count_mets_++;
        
        // update search position: 
        start = what[0].second;
    }
    
    boost::regex exprQuadJet("([a-zA-Z]*?)QuadJet([0-9]+)"); 
    start = name_.begin();    
    while ( boost::regex_search(start, end, what, exprQuadJet) ) {
        int energy = 0;
        string2int(what[2].str().c_str(),energy);
        quadjets_.insert(std::pair<int,std::string>(energy,what[1]));
        count_quadjets_++;
        
        // update search position: 
        start = what[0].second;        
    }
    
    boost::regex exprPFMHT("([a-zA-Z]*?)PFMHT([0-9]+)"); 
    start = name_.begin();
    while ( boost::regex_search(start, end, what, exprPFMHT) ) {
        int energy = 0;
        string2int(what[2].str().c_str(),energy);
        pfmht_.insert(std::pair<int,std::string>(energy,what[1]));
        count_pfmhts_++;
        
        // update search position: 
        start = what[0].second;
    }
    
    boost::regex exprHT("_HT([0-9]+)"); 
    start = name_.begin();
    while ( boost::regex_search(start, end, what, exprHT) ) {
        int energy = 0;
        string2int(what[1].str().c_str(),energy);
        ht_.insert(std::pair<int,std::string>(energy,""));
        count_hts_++;
        
        // update search position: 
        start = what[0].second;
    }
}

void HLTTauDQMFilter::setType() {
    if ( alias_ != "" ) {
        type_ = alias_;
    } else {
        if ( count_taus_ == 2 && count_muons_ == 0 && count_electrons_ == 0 && count_mets_ == 0 && count_quadjets_ == 0 && count_pfmhts_ == 0 && count_hts_ == 0 ) {
            type_ = "DoubleTau";
        } else if ( count_taus_ == 1 && count_muons_ == 0 && count_electrons_ == 0 && count_mets_ == 0 && count_quadjets_ == 0 && count_pfmhts_ == 0 && count_hts_ == 0 ) {
            std::multimap<int,std::string>::const_iterator iter = taus_.end(); --iter;
            type_ = "Single" + iter->second + "Tau";
        } else if ( count_taus_ == 1 && count_muons_ == 0 && count_electrons_ == 0 && count_mets_ == 1 && count_quadjets_ == 0 && count_pfmhts_ == 0 && count_hts_ == 0 ) {
            std::multimap<int,std::string>::const_iterator iter = taus_.end(); --iter;
            type_ = "Single" + iter->second + "Tau_MET";
        } else if ( count_taus_ == 1 && count_muons_ == 0 && count_electrons_ == 1 && count_mets_ == 0 && count_quadjets_ == 0 && count_pfmhts_ == 0 && count_hts_ == 0 ) {
            std::multimap<int,std::string>::const_iterator iter = taus_.end(); --iter;
            type_ = "Ele" + iter->second + "Tau";
        } else if ( count_taus_ == 1 && count_muons_ == 1 && count_electrons_ == 0 && count_mets_ == 0 && count_quadjets_ == 0 && count_pfmhts_ == 0 && count_hts_ == 0 ) {
            std::multimap<int,std::string>::const_iterator iter = taus_.end(); --iter;
            type_ = "Mu" + iter->second + "Tau";
        } else if ( count_taus_ == 1 && count_muons_ == 0 && count_electrons_ == 0 && count_mets_ == 0 && count_quadjets_ == 1 && count_pfmhts_ == 0 && count_hts_ == 0 ) {
            std::multimap<int,std::string>::const_iterator iter = taus_.end(); --iter;
            type_ = "QuadJet_Single" + iter->second + "Tau";
        } else if ( count_taus_ == 2 && count_muons_ == 0 && count_electrons_ == 0 && count_mets_ == 0 && count_quadjets_ == 0 && count_pfmhts_ == 1 && count_hts_ == 1 ) {
            type_ = "HT_DoubleTau_PFMHT";
        } else {
            type_ = "Unknown";
        }
    }
}

bool HLTTauDQMFilter::string2int( const char* digit, int& result ) {
    result = 0;
    
    //--- Convert each digit char and add into result.
    while (*digit >= '0' && *digit <='9') {
        result = (result * 10) + (*digit - '0');
        digit++;
    }
    
    //--- Check that there were no non-digits at end.
    if (*digit != 0) {
        return false;
    }
    
    return true;
}

std::map<int,std::string> HLTTauDQMFilter::interestingModules( HLTConfigProvider const& HLTCP ) {
    std::map<int,std::string> modules;
    if ( HLTCP.inited() ) {
        boost::regex rePFTau(".*PFTau.*");
        boost::regex rePFIsoTau(".*PFIsoTau.*");
        
        const std::vector<std::string>& moduleLabels = HLTCP.moduleLabels(name_);
        for ( std::vector<std::string>::const_iterator imodule = moduleLabels.begin(); imodule != moduleLabels.end(); ++imodule ) {
            int idx = imodule - moduleLabels.begin();
            
            if ( HLTCP.moduleType(*imodule) == "HLTLevel1GTSeed" ) {
                insertUniqueValue(modules, std::make_pair(idx,*imodule));
            } else if ( HLTCP.moduleType(*imodule) == "HLT1Tau" || HLTCP.moduleType(*imodule) == "HLT1PFTau" || HLTCP.moduleType(*imodule) == "HLT1SmartTau" ) {
                if ( boost::regex_match(*imodule, rePFTau) || boost::regex_match(*imodule, rePFIsoTau) ) {
                    insertUniqueValue(modules, std::make_pair(idx,*imodule));
                }
            } else if ( HLTCP.moduleType(*imodule) == "HLT2ElectronTau" || HLTCP.moduleType(*imodule) == "HLT2ElectronPFTau" || HLTCP.moduleType(*imodule) == "HLT2MuonTau" || HLTCP.moduleType(*imodule) == "HLT2MuonPFTau" ) {
                if ( boost::regex_match(*imodule, rePFTau) || boost::regex_match(*imodule, rePFIsoTau) ) {
                    insertUniqueValue(modules, std::make_pair(idx,*imodule));
                    
                    std::string input1 = HLTCP.modulePSet(*imodule).getParameter<edm::InputTag>("inputTag1").label();
                    int idx1 = std::find(moduleLabels.begin(), moduleLabels.end(), input1) - moduleLabels.begin();
                    
                    std::string input2 = HLTCP.modulePSet(*imodule).getParameter<edm::InputTag>("inputTag2").label();
                    int idx2 = std::find(moduleLabels.begin(), moduleLabels.end(), input2) - moduleLabels.begin();                            
                    
                    if ( HLTCP.moduleType(input1) == "HLT1Tau" || HLTCP.moduleType(input1) == "HLT1PFTau" || HLTCP.moduleType(input1) == "HLT1SmartTau" ) {
                        if ( boost::regex_match(input1, rePFTau) || boost::regex_match(input1, rePFIsoTau) ) {
                            insertUniqueValue(modules, std::make_pair(idx1,input1));
                        }
                    } else {
                        insertUniqueValue(modules, std::make_pair(idx1,input1));
                    }
                    
                    if ( HLTCP.moduleType(input2) == "HLT1Tau" || HLTCP.moduleType(input2) == "HLT1PFTau" || HLTCP.moduleType(input2) == "HLT1SmartTau" ) {
                        if ( boost::regex_match(input2, rePFTau) || boost::regex_match(input2, rePFIsoTau) ) {
                            insertUniqueValue(modules, std::make_pair(idx2,input2));
                        }
                    } else {
                        insertUniqueValue(modules, std::make_pair(idx2,input2));
                    }
                }
            }
        }
    }
    if ( alias_ != "" ) {
        for ( std::map<int,std::string>::iterator iter = modules.begin(); iter != modules.end(); ) {
            std::map<int,std::string>::iterator tempItr = iter++;
            std::string const& value = HLTCP.moduleType(tempItr->second);
            if ( value != "HLT1Tau" && value != "HLT1PFTau" && value != "HLT1SmartTau" ) {
                modules.erase(tempItr);
            }
        }
    }
    return modules;
}

bool HLTTauDQMFilter::insertUniqueValue( std::map<int,std::string>& container, std::pair<int,std::string> const& value ) {
    bool unique = true;
    for ( std::map<int,std::string>::const_iterator iter = container.begin(); iter != container.end(); ++iter ) {
        if ( iter->second == value.second ) {
            unique = false;
            break;
        }
    }
    if ( unique ) container.insert(value);
    return unique;
}

std::vector<edm::ParameterSet> HLTTauDQMFilter::getPSets( HLTConfigProvider const& HLTCP ) {
    std::vector<edm::ParameterSet> psets;
    std::map<int,std::string> modules = interestingModules(HLTCP);
    
    boost::regex reElectron(".*Electron.*");
    boost::regex reMuon(".*Muon.*");
    
    for ( std::map<int,std::string>::const_iterator imodule = modules.begin(); imodule != modules.end(); ++imodule ) {
        edm::ParameterSet tmp;
        tmp.addUntrackedParameter<edm::InputTag>( "FilterName", edm::InputTag(imodule->second,"",hltTauDQMProcess_) );
        
        if ( HLTCP.moduleType(imodule->second) == "HLTLevel1GTSeed" ) {
            tmp.addUntrackedParameter<double>( "MatchDeltaR", L1MatchDeltaR_ );
            if ( count_taus_ == 2 && count_muons_ == 0 && count_electrons_ == 0 ) { //DoubleTau
                tmp.addUntrackedParameter<unsigned int>( "NTriggeredTaus", 2 );
                tmp.addUntrackedParameter<int>( "TauType", trigger::TriggerL1TauJet );
                tmp.addUntrackedParameter<unsigned int>( "NTriggeredLeptons", 0 );
                tmp.addUntrackedParameter<int>( "LeptonType", 0 );
            } else if ( count_taus_ == 1 && count_muons_ == 0 && count_electrons_ == 0 ) { //SingleTau
                tmp.addUntrackedParameter<unsigned int>( "NTriggeredTaus", 1 );
                tmp.addUntrackedParameter<int>( "TauType", trigger::TriggerL1TauJet );
                tmp.addUntrackedParameter<unsigned int>( "NTriggeredLeptons", 0 );
                tmp.addUntrackedParameter<int>( "LeptonType", 0 );
            } else if ( count_taus_ == 1 && count_muons_ == 0 && count_electrons_ == 1 ) { //EleTau
                tmp.addUntrackedParameter<unsigned int>( "NTriggeredTaus", 0 );
                tmp.addUntrackedParameter<int>( "TauType", 0 );
                tmp.addUntrackedParameter<unsigned int>( "NTriggeredLeptons", 1 );
                tmp.addUntrackedParameter<int>( "LeptonType", trigger::TriggerL1IsoEG );
            } else if ( count_taus_ == 1 && count_muons_ == 1 && count_electrons_ == 0 ) { //MuTau
                tmp.addUntrackedParameter<unsigned int>( "NTriggeredTaus", 0 );
                tmp.addUntrackedParameter<int>( "TauType", 0 );
                tmp.addUntrackedParameter<unsigned int>( "NTriggeredLeptons", 1 );
                tmp.addUntrackedParameter<int>( "LeptonType", trigger::TriggerL1Mu );
            } else { //Unknown
                tmp.addUntrackedParameter<unsigned int>( "NTriggeredTaus", 0 );
                tmp.addUntrackedParameter<int>( "TauType", 0 );
                tmp.addUntrackedParameter<unsigned int>( "NTriggeredLeptons", 0 );
                tmp.addUntrackedParameter<int>( "LeptonType", 0 );
            }
        } else if ( HLTCP.moduleType(imodule->second) == "HLT1Tau" || HLTCP.moduleType(imodule->second) == "HLT1PFTau" || HLTCP.moduleType(imodule->second) == "HLT1SmartTau" ) {
            tmp.addUntrackedParameter<double>( "MatchDeltaR", HLTMatchDeltaR_ );
            tmp.addUntrackedParameter<unsigned int>( "NTriggeredTaus", count_taus_ );
            tmp.addUntrackedParameter<int>( "TauType", trigger::TriggerTau );
            tmp.addUntrackedParameter<unsigned int>( "NTriggeredLeptons", 0 );
            tmp.addUntrackedParameter<int>( "LeptonType", 0 );
        } else if ( HLTCP.moduleType(imodule->second) == "HLT2ElectronTau" || HLTCP.moduleType(imodule->second) == "HLT2ElectronPFTau" ) {
            tmp.addUntrackedParameter<double>( "MatchDeltaR", HLTMatchDeltaR_ );
            tmp.addUntrackedParameter<unsigned int>( "NTriggeredTaus", count_taus_ );
            tmp.addUntrackedParameter<int>( "TauType", trigger::TriggerTau );
            tmp.addUntrackedParameter<unsigned int>( "NTriggeredLeptons", count_electrons_ );
            tmp.addUntrackedParameter<int>( "LeptonType", trigger::TriggerElectron );
        } else if ( HLTCP.moduleType(imodule->second) == "HLT2MuonTau" || HLTCP.moduleType(imodule->second) == "HLT2MuonPFTau" ) {
            tmp.addUntrackedParameter<double>( "MatchDeltaR", HLTMatchDeltaR_ );
            tmp.addUntrackedParameter<unsigned int>( "NTriggeredTaus", count_taus_ );
            tmp.addUntrackedParameter<int>( "TauType", trigger::TriggerTau );
            tmp.addUntrackedParameter<unsigned int>( "NTriggeredLeptons", count_muons_ );
            tmp.addUntrackedParameter<int>( "LeptonType", trigger::TriggerMuon );
        } else if ( boost::regex_match(HLTCP.moduleType(imodule->second), reElectron) ) {
            tmp.addUntrackedParameter<double>( "MatchDeltaR", HLTMatchDeltaR_ );
            tmp.addUntrackedParameter<unsigned int>( "NTriggeredTaus", 0 );
            tmp.addUntrackedParameter<int>( "TauType", 0 );
            tmp.addUntrackedParameter<unsigned int>( "NTriggeredLeptons", count_electrons_ );
            tmp.addUntrackedParameter<int>( "LeptonType", trigger::TriggerElectron );
        } else if ( boost::regex_match(HLTCP.moduleType(imodule->second), reMuon) ) {
            tmp.addUntrackedParameter<double>( "MatchDeltaR", HLTMatchDeltaR_ );
            tmp.addUntrackedParameter<unsigned int>( "NTriggeredTaus", 0 );
            tmp.addUntrackedParameter<int>( "TauType", 0 );
            tmp.addUntrackedParameter<unsigned int>( "NTriggeredLeptons", count_muons_ );
            tmp.addUntrackedParameter<int>( "LeptonType", trigger::TriggerMuon );
        }
        psets.push_back(tmp);
    }
    
    return psets;
}

edm::ParameterSet HLTTauDQMFilter::getRefPSet() {
    edm::ParameterSet tmp;
    tmp.addUntrackedParameter<unsigned int>( "NTriggeredTaus", NReferenceTaus() );
    tmp.addUntrackedParameter<unsigned int>( "NTriggeredLeptons", NReferenceLeptons() );
    
    return tmp;
}

edm::ParameterSet HLTTauDQMFilter::getSummaryPSet( HLTConfigProvider const& HLTCP ) {
    edm::ParameterSet tmp;
    std::vector<edm::ParameterSet> modules = getPSets(HLTCP);
    if (modules.size() > 0) {
        tmp = modules.back();
        tmp.addUntrackedParameter<std::string>( "Alias", type() );
        tmp.addUntrackedParameter<int>( "TauType", 15 );
        if ( tmp.getUntrackedParameter<int>( "LeptonType" ) == trigger::TriggerElectron ) {
            tmp.addUntrackedParameter<int>( "LeptonType", 11 );
        } else if ( tmp.getUntrackedParameter<int>( "LeptonType" ) == trigger::TriggerMuon ) {
            tmp.addUntrackedParameter<int>( "LeptonType", 13 );
        }
    }
    return tmp;
}
