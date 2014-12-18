#ifndef RazorVarAnalyzer_h
#define RazorVarAnalyzer_h

#include "DQM/DataScouting/interface/ScoutingAnalyzerBase.h"

#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

class RazorVarAnalyzer : public ScoutingAnalyzerBase {
  public:
    explicit RazorVarAnalyzer( const edm::ParameterSet &  ) ;
    virtual ~RazorVarAnalyzer() ;
    void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
    virtual void analyze( const edm::Event & , const edm::EventSetup &  );
  private: 
    edm::InputTag m_eleCollectionTag;    
    edm::InputTag m_jetCollectionTag;
    edm::InputTag m_muCollectionTag;
    edm::InputTag m_razorVarCollectionTag;

    //inclusive histograms by jet number
    MonitorElement * m_rsqMRFullyInc;
    MonitorElement * m_rsqMRInc4J;
    MonitorElement * m_rsqMRInc6J;
    MonitorElement * m_rsqMRInc8J;
    MonitorElement * m_rsqMRInc10J;
    MonitorElement * m_rsqMRInc12J;
    MonitorElement * m_rsqMRInc14J;

    //per box histograms
    MonitorElement * m_rsqMREleMu;
    MonitorElement * m_rsqMRMuMu;
    MonitorElement * m_rsqMREleEle;
    MonitorElement * m_rsqMRMu;
    MonitorElement * m_rsqMREle;
    MonitorElement * m_rsqMRHad;

    //now per box multijet
    MonitorElement * m_rsqMRMuMJ;
    MonitorElement * m_rsqMREleMJ;
    MonitorElement * m_rsqMRHadMJ;

    //define Token(-s)
    edm::EDGetTokenT<reco::CaloJetCollection> m_jetCollectionTagToken_;
    edm::EDGetTokenT<std::vector<reco::RecoChargedCandidate> > m_muCollectionTagToken_;
    edm::EDGetTokenT<reco::ElectronCollection> m_eleCollectionTagToken_;
    edm::EDGetTokenT<std::vector<double> > m_razorVarCollectionTagToken_;
};
#endif
