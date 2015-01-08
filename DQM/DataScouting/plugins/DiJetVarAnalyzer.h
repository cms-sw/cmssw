#ifndef DiJetVarAnalyzer_h
#define DiJetVarAnalyzer_h

#include "DQM/DataScouting/interface/ScoutingAnalyzerBase.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionData.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionEvaluator.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

#include "TLorentzVector.h"
#include <vector>
#include <cmath>

class DiJetVarAnalyzer : public ScoutingAnalyzerBase {
  public:
    explicit DiJetVarAnalyzer( const edm::ParameterSet &  ) ;
    virtual ~DiJetVarAnalyzer() ;
    void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
    virtual void analyze( const edm::Event & , const edm::EventSetup &  );
  private: 
    edm::InputTag jetCollectionTag_;
    edm::InputTag widejetsCollectionTag_;
    edm::InputTag metCollectionTag_;
    edm::InputTag metCleanCollectionTag_;
    edm::InputTag hltInputTag_;

    unsigned int     numwidejets_;    
    double  etawidejets_;
    double  ptwidejets_;
    double  detawidejets_;
    double  dphiwidejets_;
    double  maxEMfraction_;
    double  maxHADfraction_;

    // trigger conditions
    triggerExpression::Evaluator * HLTpathMain_;
    triggerExpression::Evaluator * HLTpathMonitor_;
    // cache some data from the Event for faster access by the trigger conditions
    triggerExpression::Data triggerConfiguration_;
    //1D histograms
    MonitorElement * m_cutFlow;
    MonitorElement * m_MjjWide_finalSel;
    MonitorElement * m_MjjWide_finalSel_varbin;
    MonitorElement * m_MjjWide_finalSel_WithoutNoiseFilter;
    MonitorElement * m_MjjWide_finalSel_WithoutNoiseFilter_varbin;    
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

    MonitorElement * m_metCases;
    MonitorElement * m_metDiff;
    MonitorElement * m_metCaseNoMetClean;

    MonitorElement * m_HT_inclusive;
    MonitorElement * m_HT_finalSel;

    //2D histograms
    MonitorElement * m_DetajjVsMjjWide;
    MonitorElement * m_DetajjVsMjjWide_rebin;

    MonitorElement * m_metVSmetclean;

    //define Token(-s)
    edm::EDGetTokenT<reco::CaloJetCollection> jetCollectionTagToken_;
    edm::EDGetTokenT<std::vector<math::PtEtaPhiMLorentzVector> > widejetsCollectionTagToken_;
    edm::EDGetTokenT<reco::CaloMETCollection> metCollectionTagToken_;
    edm::EDGetTokenT<reco::CaloMETCollection> metCleanCollectionTagToken_;
};
#endif
