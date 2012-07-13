#ifndef DiJetVarAnalyzer_h
#define DiJetVarAnalyzer_h


#include "DQM/DataScouting/interface/ScoutingAnalyzerBase.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionData.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionEvaluator.h"

#include "TLorentzVector.h"
#include <vector>

class DiJetVarAnalyzer : public ScoutingAnalyzerBase
 {

  public:

    explicit DiJetVarAnalyzer( const edm::ParameterSet &  ) ;
    virtual ~DiJetVarAnalyzer() ;

    virtual void analyze( const edm::Event & , const edm::EventSetup &  );
    virtual void beginRun( const edm::Run &, const edm::EventSetup & );
    virtual void endRun( edm::Run const &, edm::EventSetup const & ) ;

    virtual void bookMEs();

  private: 

    edm::InputTag jetCollectionTag_;
    //edm::InputTag dijetVarCollectionTag_;
    edm::InputTag widejetsCollectionTag_;
    edm::InputTag hltInputTag_;

    unsigned int     numwidejets_;    
    double  etawidejets_;
    double  ptwidejets_;
    double  detawidejets_;
    double  dphiwidejets_;

    // trigger conditions
    triggerExpression::Evaluator * HLTpathMain_;
    triggerExpression::Evaluator * HLTpathMonitor_;
    
    // cache some data from the Event for faster access by the trigger conditions
    triggerExpression::Data triggerConfiguration_;
    
    //1D histograms
    MonitorElement * m_cutFlow;

    MonitorElement * m_MjjWide_finalSel;
    MonitorElement * m_MjjWide_finalSel_varbin;
    MonitorElement * m_MjjWide_deta_0p0_0p5;
    MonitorElement * m_MjjWide_deta_0p5_1p0;
    MonitorElement * m_MjjWide_deta_1p0_1p5;
    MonitorElement * m_MjjWide_deta_1p5_2p0;
    MonitorElement * m_MjjWide_deta_2p0_2p5;
    MonitorElement * m_MjjWide_deta_2p5_3p0;
    MonitorElement * m_MjjWide_deta_3p0_inf;

    MonitorElement * m_DetajjWide_finalSel;
    MonitorElement * m_DetajjWide;

    MonitorElement * m_DphijjWide_finalSel;

    MonitorElement * m_selJets_pt; 
    MonitorElement * m_selJets_eta;
    MonitorElement * m_selJets_phi;
    MonitorElement * m_selJets_hadEnergyFraction;
    MonitorElement * m_selJets_emEnergyFraction;
    MonitorElement * m_selJets_towersArea;

    MonitorElement * m_MjjWide_den_NOdeta;
    MonitorElement * m_MjjWide_num_NOdeta;
    MonitorElement * m_MjjWide_den_detaL4;
    MonitorElement * m_MjjWide_num_detaL4;
    MonitorElement * m_MjjWide_den_detaL3;
    MonitorElement * m_MjjWide_num_detaL3;
    MonitorElement * m_MjjWide_den_detaL2;
    MonitorElement * m_MjjWide_num_detaL2;
    MonitorElement * m_MjjWide_den;
    MonitorElement * m_MjjWide_num;

    //2D histograms
    MonitorElement * m_DetajjVsMjjWide;
    MonitorElement * m_DetajjVsMjjWide_rebin;
 } ;

#endif
