#ifndef RazorVarAnalyzer_h
#define RazorVarAnalyzer_h


#include "DQM/DataScouting/interface/ScoutingAnalyzerBase.h"

class RazorVarAnalyzer : public ScoutingAnalyzerBase
 {

  public:

    explicit RazorVarAnalyzer( const edm::ParameterSet &  ) ;
    virtual ~RazorVarAnalyzer() ;

    virtual void analyze( const edm::Event & , const edm::EventSetup &  );
    
    virtual void endRun( edm::Run const &, edm::EventSetup const & ) ;

    virtual void bookMEs();

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


 } ;

#endif
