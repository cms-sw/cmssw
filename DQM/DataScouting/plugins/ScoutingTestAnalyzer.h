#ifndef ScoutingTestAnalyzer_h
#define ScoutingTestAnalyzer_h

// This class is used to test the functionalities of the package


#include "DQM/DataScouting/interface/ScoutingAnalyzerBase.h"

class ScoutingTestAnalyzer : public ScoutingAnalyzerBase
 {

  public:

    explicit ScoutingTestAnalyzer( const edm::ParameterSet &  ) ;
    virtual ~ScoutingTestAnalyzer() ;

    virtual void analyze( const edm::Event & , const edm::EventSetup &  );
    
    virtual void endRun( edm::Run const &, edm::EventSetup const & ) ;

    virtual void bookMEs();

  private: 
    
    // histograms
    edm::InputTag m_pfJetsCollectionTag;
    MonitorElement * m_jetPt;
    MonitorElement * m_jetEtaPhi;

 } ;

#endif

