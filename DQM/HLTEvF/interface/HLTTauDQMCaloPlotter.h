// Original Author:  Michail Bachtis
// Created:  Sun Jan 20 20:10:02 CST 2008
// University of Wisconsin-Madison

#ifndef HLTTauDQMCaloPlotter_h
#define HLTTauDQMCaloPlotter_h

#include "DQM/HLTEvF/interface/HLTTauDQMPlotter.h"

#include "DataFormats/TauReco/interface/L2TauInfoAssociation.h"

class HLTTauDQMCaloPlotter : public HLTTauDQMPlotter {
public:
    HLTTauDQMCaloPlotter( const edm::ParameterSet&, int, int, int, double, bool, double, std::string );
    ~HLTTauDQMCaloPlotter();
    const std::string name() { return name_; }
    void analyze( const edm::Event&, const edm::EventSetup&, const std::map<int,LVColl>& );
    
private:
    //Parameters to read
    std::vector<edm::InputTag> l2preJets_;
    edm::InputTag l2TauInfoAssoc_; //Path to analyze
    edm::InputTag met_; //Handle to missing Et 
    bool doRef_; //DoReference Analysis
    
    //Select if you want match or not
    double matchDeltaRMC_;
    edm::InputTag l2Isolated_; //Path to analyze
    
    //Histogram Limits
    
    double EtMax_;
    int NPtBins_;
    int NEtaBins_;
    int NPhiBins_;
    
    //Monitor elements main
    MonitorElement* preJetEt;
    MonitorElement* preJetEta;
    MonitorElement* preJetPhi;
    
    MonitorElement* jetEt;
    MonitorElement* jetEta;
    MonitorElement* jetPhi;
    
    MonitorElement* isoJetEt;
    MonitorElement* isoJetEta;
    MonitorElement* isoJetPhi;
    
    MonitorElement* jetEtRes;
    
    MonitorElement* ecalIsolEt;
    MonitorElement* hcalIsolEt;
    
    MonitorElement* seedEcalEt;
    MonitorElement* seedHcalEt;
    
    MonitorElement* ecalClusterEtaRMS;
    MonitorElement* ecalClusterPhiRMS;
    MonitorElement* ecalClusterDeltaRRMS;
    MonitorElement* nEcalClusters;
    
    MonitorElement* hcalClusterEtaRMS;
    MonitorElement* hcalClusterPhiRMS;
    MonitorElement* hcalClusterDeltaRRMS;
    MonitorElement* nHcalClusters;
    
    MonitorElement* recoEtEffNum;
    MonitorElement* recoEtEffDenom;
    MonitorElement* recoEtaEffNum;
    MonitorElement* recoEtaEffDenom;
    MonitorElement* recoPhiEffNum;
    MonitorElement* recoPhiEffDenom;
    
    MonitorElement* isoEtEffNum;
    MonitorElement* isoEtEffDenom;
    MonitorElement* isoEtaEffNum;
    MonitorElement* isoEtaEffDenom;
    MonitorElement* isoPhiEffNum;
    MonitorElement* isoPhiEffDenom;
    
    bool matchJet( const reco::Jet&, const reco::CaloJetCollection& );//See if this Jet Is Matched
    std::pair<bool,reco::CaloJet> inverseMatch( const LV&, const reco::CaloJetCollection& );//See if this Jet Is Matched
    
    class SorterByPt {
    public:
        SorterByPt() {}
        ~SorterByPt() {}
        bool operator() (reco::CaloJet jet1 , reco::CaloJet jet2) {
            return jet1.pt() > jet2.pt();
        }
    };
};
#endif
