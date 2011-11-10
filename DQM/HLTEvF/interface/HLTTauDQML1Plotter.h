/* HLTTau Path Analyzer
 Michail Bachtis
 University of Wisconsin - Madison
 bachtis@hep.wisc.edu
 */

#ifndef HLTTauDQML1Plotter_h
#define HLTTauDQML1Plotter_h

#include "DQM/HLTEvF/interface/HLTTauDQMPlotter.h"

#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"

class HLTTauDQML1Plotter : public HLTTauDQMPlotter {
public:
    HLTTauDQML1Plotter( const edm::ParameterSet&, int, int, int, double, bool, double, std::string );
    ~HLTTauDQML1Plotter();
    const std::string name() { return name_; }
    void analyze( const edm::Event&, const edm::EventSetup&, const std::map<int,LVColl>& );
    
private:
    void endJob() ;
    
    //The filters
    edm::InputTag l1ExtraTaus_;
    edm::InputTag l1ExtraJets_;
    edm::InputTag l1ExtraElectrons_;
    edm::InputTag l1ExtraMuons_;
    
    bool doRefAnalysis_;
    double matchDeltaR_;
    
    double maxEt_;
    int binsEt_;
    int binsEta_;
    int binsPhi_;
    
    //MonitorElements general
    MonitorElement* l1tauEt_;
    MonitorElement* l1tauEta_;
    MonitorElement* l1tauPhi_;
    
    MonitorElement* l1jetEt_;
    MonitorElement* l1jetEta_;
    MonitorElement* l1jetPhi_;
    
    MonitorElement* l1electronEt_;
    MonitorElement* l1electronEta_;
    MonitorElement* l1electronPhi_;
    
    MonitorElement* l1muonEt_;
    MonitorElement* l1muonEta_;
    MonitorElement* l1muonPhi_;
    
    //Monitor Elements for matching
    MonitorElement* inputEvents_;
    
    MonitorElement* l1tauEtRes_;
    
    MonitorElement* l1tauEtEffNum_;
    MonitorElement* l1tauEtEffDenom_;
    
    MonitorElement* l1tauEtaEffNum_;
    MonitorElement* l1tauEtaEffDenom_;
    
    MonitorElement* l1tauPhiEffNum_;
    MonitorElement* l1tauPhiEffDenom_;
    
    MonitorElement* l1jetEtEffNum_;
    MonitorElement* l1jetEtEffDenom_;
    
    MonitorElement* l1jetEtaEffNum_;
    MonitorElement* l1jetEtaEffDenom_;
    
    MonitorElement* l1jetPhiEffNum_;
    MonitorElement* l1jetPhiEffDenom_;
    
    MonitorElement* l1doubleTauPath_;
    MonitorElement* l1electronTauPath_;
    MonitorElement* l1muonTauPath_;
    
    MonitorElement* firstTauEt_;
    MonitorElement* secondTauEt_;
    
    struct ComparePt {
        bool operator() (LV l1,LV l2) {
            return l1.pt() > l2.pt();
        }
    };
    
    ComparePt ptSort;
};
#endif
