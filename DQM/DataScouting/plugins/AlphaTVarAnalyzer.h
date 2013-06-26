#ifndef AlphaTVarAnalyzer_h
#define AlphaTVarAnalyzer_h


#include "DQM/DataScouting/interface/ScoutingAnalyzerBase.h"

class AlphaTVarAnalyzer : public ScoutingAnalyzerBase
 {

  public:

    explicit AlphaTVarAnalyzer( const edm::ParameterSet &  ) ;
    virtual ~AlphaTVarAnalyzer() ;

    virtual void analyze( const edm::Event & , const edm::EventSetup &  );
    
    virtual void endRun( edm::Run const &, edm::EventSetup const & ) ;

    virtual void bookMEs();

  private: 

    edm::InputTag m_jetCollectionTag;
    edm::InputTag m_alphaTVarCollectionTag;

    //inclusive histograms by jet number
    MonitorElement * m_HTAlphaT;
    MonitorElement * m_HTAlphaTg0p55;
    MonitorElement * m_HTAlphaTg0p60;

 } ;

#endif
