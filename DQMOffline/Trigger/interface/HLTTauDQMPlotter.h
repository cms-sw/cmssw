#ifndef HLTTauDQMPlotter_h
#define HLTTauDQMPlotter_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

//Include DQM core
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "Math/GenVector/VectorUtil.h"

typedef math::XYZTLorentzVectorD LV;
typedef std::vector<LV> LVColl;

//Virtual base class for HLT-Tau-DQM Plotters
class HLTTauDQMPlotter {
public:
    HLTTauDQMPlotter();
    virtual ~HLTTauDQMPlotter();
    virtual const std::string name() = 0;
    bool isValid() const { return validity_; }

protected:
    //Helper functions
    std::pair<bool,LV> match( const LV&, const LVColl&, double );    
    std::string triggerTag();
    
    class FilterObject {
    public:
        FilterObject( const edm::ParameterSet& ps );
        virtual ~FilterObject();
        
        edm::InputTag getFilterName() const { return FilterName_; }
        double getMatchDeltaR() const { return MatchDeltaR_; }
        unsigned int getNTriggeredTaus() const { return NTriggeredTaus_; }
        int getTauType() const { return TauType_; }
        unsigned int getNTriggeredLeptons() const { return NTriggeredLeptons_; }
        int getLeptonType() const { return LeptonType_; }
        int leptonId();
        std::string getAlias() const { return Alias_; }
        bool isValid() const { return validity_; }
        
    private:
        edm::InputTag FilterName_;
        double MatchDeltaR_;
        unsigned int NTriggeredTaus_;
        int TauType_;
        unsigned int NTriggeredLeptons_;
        int LeptonType_;
        std::string Alias_;
        bool validity_;
    };
    
    //DQM store service
    DQMStore* store_;
    
    //Name of the Plotter
    std::string name_;
    
    //DQM folders
    std::string dqmBaseFolder_;
    std::string triggerTag_;
    std::string triggerTagAlias_;
    
    //Validity check
    bool validity_;
};
#endif
