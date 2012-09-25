#ifndef DiJetPairsVarAnalyzer_h
#define DiJetPairsVarAnalyzer_h


#include "DQM/DataScouting/interface/ScoutingAnalyzerBase.h"

class DiJetPairsVarAnalyzer : public ScoutingAnalyzerBase
 {

  public:

    explicit DiJetPairsVarAnalyzer( const edm::ParameterSet &  ) ;
    virtual ~DiJetPairsVarAnalyzer() ;
    virtual void analyze( const edm::Event & , const edm::EventSetup &  );
    virtual void endRun( edm::Run const &, edm::EventSetup const & ) ;
    virtual void bookMEs();

  private: 

    double jetPtCut_,htCut_,delta_;
    edm::InputTag jetPtCollectionTag_;
    edm::InputTag dijetPtCollectionTag_;  
    edm::InputTag dijetdRCollectionTag_;
    edm::InputTag dijetMassCollectionTag_;

    MonitorElement * me_fourthJetPt;
    MonitorElement * me_Ht;
    MonitorElement * me_Njets;
    MonitorElement * me_MassDiff;
    MonitorElement * me_AvgDiJetMass;
    MonitorElement * me_DeltavsAvgMass;

 } ;

#endif
