// Original Author:  Eduardo Luigi
//         Created:  Sun Jan 20 20:10:02 CST 2008

#ifndef HLTTauDQMTrkPlotter_h
#define HLTTauDQMTrkPlotter_h

#include "DQM/HLTEvF/interface/HLTTauDQMPlotter.h"

#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/TrackReco/interface/Track.h"

class HLTTauDQMTrkPlotter : public HLTTauDQMPlotter {
public:
    HLTTauDQMTrkPlotter( const edm::ParameterSet&, int, int, int, double, bool, double, std::string );
    ~HLTTauDQMTrkPlotter();
    const std::string name() { return name_; }
    void analyze( const edm::Event&, const edm::EventSetup&, const std::map<int,LVColl>& );
    
private:
    bool matchJet( const reco::Jet&, const reco::CaloJetCollection& ); 
    
    //Parameters to read
    edm::InputTag jetTagSrc_;
    edm::InputTag isolJets_;
    
    //Output file
    std::string type_;
    double mcMatch_;
    //Monitor elements main
    MonitorElement* jetEt;
    MonitorElement* jetEta;
    MonitorElement* jetPhi;
    
    MonitorElement* isoJetEt;
    MonitorElement* isoJetEta;
    MonitorElement* isoJetPhi;
    
    MonitorElement* nPxlTrksInL25Jet;
    MonitorElement* nQPxlTrksInL25Jet;
    MonitorElement* signalLeadTrkPt;
    MonitorElement* hasLeadTrack;
    
    MonitorElement* EtEffNum;
    MonitorElement* EtEffDenom;
    MonitorElement* EtaEffNum;
    MonitorElement* EtaEffDenom;
    MonitorElement* PhiEffNum;
    MonitorElement* PhiEffDenom;
    
    bool doRef_;
    
    //Histogram Limits
    double EtMax_;
    int NPtBins_;
    int NEtaBins_;
    int NPhiBins_;
};
#endif
