#ifndef RazorVarAnalyzer_h
#define RazorVarAnalyzer_h

#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "DQM/DataScouting/interface/ScoutingAnalyzerBase.h"

#include <string>

class RazorVarAnalyzer : public ScoutingAnalyzerBase
 {

  public:

    explicit RazorVarAnalyzer( const edm::ParameterSet &  ) ;
    virtual ~RazorVarAnalyzer() ;

    virtual void analyze( const edm::Event & , const edm::EventSetup &  );
    bool triggerRegexp(const edm::TriggerResults&, const edm::TriggerNames&, const std::string&) const;
    
    virtual void endRun( edm::Run const &, edm::EventSetup const & ) ;

    virtual void bookMEs();

  private: 

    edm::InputTag m_eleCollectionTag;    
    edm::InputTag m_jetCollectionTag;
    edm::InputTag m_muCollectionTag;
    //for Noise filtering
    edm::InputTag m_metCollectionTag;
    edm::InputTag m_metCleanCollectionTag;
    //now the trigger
    edm::InputTag m_triggerCollectionTag;

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

    //calibration histograms
    MonitorElement * m_rsqMRFullyIncNoise;
    MonitorElement * m_rsqMRFullyIncLevel1;
    
    //turn on histograms
    MonitorElement * m_rsqMRMuLevel1;
    MonitorElement * m_rsqMREleLevel1;
    MonitorElement * m_rsqMRHadLevel1;

    MonitorElement * m_rsqMRMuHLT;
    MonitorElement * m_rsqMREleHLT;
    MonitorElement * m_rsqMRHadHLT;



 } ;

#endif
