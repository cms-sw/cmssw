#ifndef HLTTauDQMFilter_h
#define HLTTauDQMFilter_h

#include <iostream>
#include <map>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

class HLTTauDQMFilter {
public:
    HLTTauDQMFilter( std::string const& name, int initialPrescale, std::string hltTauDQMProcess, double L1MatchDeltaR, double HLTMatchDeltaR );
    HLTTauDQMFilter( std::string const& name, std::string const& alias, int initialPrescale, std::string hltTauDQMProcess, double L1MatchDeltaR, double HLTMatchDeltaR, int nRefTaus = -1, int nRefElectrons = -1, int nRefMuons = -1 );
    virtual ~HLTTauDQMFilter();
    std::string name() const { return name_; }
    int initialPrescale() const { return initialPrescale_; }
    void print();
    std::string type() const { return type_; }
    int leadingTau() const;
    int leadingMuon() const;
    int leadingElectron() const;
    int leadingMET() const;
    int leadingQuadJet() const;
    int leadingPFMHT() const;
    int leadingHT() const;
    std::map<int,std::string> interestingModules( HLTConfigProvider const& HLTCP );
    std::vector<edm::ParameterSet> getPSets( HLTConfigProvider const& HLTCP );
    edm::ParameterSet getRefPSet();
    edm::ParameterSet getSummaryPSet( HLTConfigProvider const& HLTCP );
    int NReferenceTaus() const { return count_taus_; }
    int NReferenceLeptons() const { return count_electrons_+count_muons_; }
    HLTTauDQMFilter const& select( HLTTauDQMFilter const& filter, bool& swapped );

    
private:
    void regexSearch();
    void setType();
    bool string2int( const char* digit, int& result );
    bool insertUniqueValue( std::map<int,std::string>& container, std::pair<int,std::string> const& value );
    
    std::multimap<int,std::string> taus_;
    unsigned int count_taus_;
    std::multimap<int,std::string> muons_;
    unsigned int count_muons_;
    std::multimap<int,std::string> electrons_;
    unsigned int count_electrons_;
    std::multimap<int,std::string> mets_;
    unsigned int count_mets_;
    std::multimap<int,std::string> quadjets_;
    unsigned int count_quadjets_;
    std::multimap<int,std::string> pfmht_;
    unsigned int count_pfmhts_;
    std::multimap<int,std::string> ht_;
    unsigned int count_hts_;
    
    std::string name_;
    std::string alias_;
    std::string type_;
    int initialPrescale_;
    std::string hltTauDQMProcess_;
    double L1MatchDeltaR_;
    double HLTMatchDeltaR_;
};

#endif
