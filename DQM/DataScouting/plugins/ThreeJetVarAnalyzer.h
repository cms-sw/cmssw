#ifndef ThreeJetVarAnalyzer_h
#define ThreeJetVarAnalyzer_h


#include "DQM/DataScouting/interface/ScoutingAnalyzerBase.h"

class ThreeJetVarAnalyzer : public ScoutingAnalyzerBase
 {

  public:

    explicit ThreeJetVarAnalyzer( const edm::ParameterSet &  ) ;
    virtual ~ThreeJetVarAnalyzer() ;
    virtual void analyze( const edm::Event & , const edm::EventSetup &  );
    virtual void endRun( edm::Run const &, edm::EventSetup const & ) ;
    virtual void bookMEs();

  private: 

    double jetPtCut_,htCut_,delta_;
    edm::InputTag jetPtCollectionTag_;
    edm::InputTag tripPtCollectionTag_;
    edm::InputTag tripMassCollectionTag_;

    //inclusive histograms by jet number
    MonitorElement * me_sixthJetPt;
    MonitorElement * me_Ht;
    MonitorElement * me_Njets;
    MonitorElement * me_TripMass;
    MonitorElement * me_TripMassvsTripPt;

 } ;

#endif
