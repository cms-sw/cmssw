#ifndef DiJetVarAnalyzer_h
#define DiJetVarAnalyzer_h


#include "DQM/DataScouting/interface/ScoutingAnalyzerBase.h"

#include "TLorentzVector.h"
#include <vector>

class DiJetVarAnalyzer : public ScoutingAnalyzerBase
 {

  public:

    explicit DiJetVarAnalyzer( const edm::ParameterSet &  ) ;
    virtual ~DiJetVarAnalyzer() ;

    virtual void analyze( const edm::Event & , const edm::EventSetup &  );
    
    virtual void endRun( edm::Run const &, edm::EventSetup const & ) ;

    virtual void bookMEs();

  private: 

    edm::InputTag jetCollectionTag_;
    //edm::InputTag dijetVarCollectionTag_;
    edm::InputTag widejetsCollectionTag_;

    unsigned int     numwidejets_;    
    double  etawidejets_;
    double  ptwidejets_;
    double  detawidejets_;
    double  dphiwidejets_;

    //1D histograms
    MonitorElement * m_cutFlow;

    MonitorElement * m_MjjWide;
    MonitorElement * m_DetajjWide;
    MonitorElement * m_DphijjWide;

    //2D histograms
    MonitorElement * m_DetajjVsMjjWide;
 } ;

#endif
