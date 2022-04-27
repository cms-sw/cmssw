#ifndef JetAnalyzer_H
#define JetAnalyzer_H

/** \class JetMETAnalyzer
 *
 *  DQM jetMET analysis monitoring
 *
 *  \author F. Chlebana - Fermilab
 *          K. Hatakeyama - Rockefeller University
 *
 *          Jan. '14: modified by
 *
 *          M. Artur Weber
 *          R. Schoefbeck
 *          V. Sordini
 */

#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/JPTJet.h"
#include "DataFormats/JetReco/interface/JPTJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"
#include "RecoJets/JetProducers/interface/JetIDHelper.h"
#include "DataFormats/MuonReco/interface/Muon.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/PatCandidates/interface/MET.h"

#include "DQMOffline/JetMET/interface/JetMETDQMDCSFilter.h"

#include "DataFormats/BTauReco/interface/CATopJetTagInfo.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "PhysicsTools/SelectorUtils/interface/JetIDSelectionFunctor.h"
#include "PhysicsTools/SelectorUtils/interface/PFJetIDSelectionFunctor.h"
#include "DataFormats/JetReco/interface/PileupJetIdentifier.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include <map>
#include <string>

//namespace jetAnalysis {
//class TrackPropagatorToCalo;
//class StripSignalOverNoiseCalculator;
//}

class JetAnalyzer : public DQMEDAnalyzer {
public:
  /// Constructor
  JetAnalyzer(const edm::ParameterSet&);

  /// Destructor
  ~JetAnalyzer() override;

  /// Inizialize parameters for histo binning
  //  void beginJob(void);
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  /// Initialize run-based parameters
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;

private:
  // ----------member data ---------------------------
  static bool jetSortingRule(reco::Jet x, reco::Jet y) { return x.pt() > y.pt(); }

  //try to put one collection as start
  edm::InputTag mInputCollection_;
  edm::InputTag theTriggerResultsLabel_;

  std::string jetType_;

  edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
  edm::EDGetTokenT<std::vector<reco::Vertex>> vertexToken_;
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> gtToken_;
  edm::EDGetTokenT<reco::CaloJetCollection> caloJetsToken_;
  edm::EDGetTokenT<reco::PFJetCollection> pfJetsToken_;

  edm::EDGetTokenT<reco::PFMETCollection> pfMetToken_;
  edm::EDGetTokenT<reco::CaloMETCollection> caloMetToken_;
  edm::EDGetTokenT<pat::METCollection> patMetToken_;

  edm::EDGetTokenT<reco::MuonCollection> MuonsToken_;
  edm::EDGetTokenT<pat::JetCollection> patJetsToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> mvaFullPUDiscriminantToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> cutBasedPUDiscriminantToken_;
  edm::EDGetTokenT<edm::ValueMap<int>> cutBasedPUIDToken_;
  edm::EDGetTokenT<edm::ValueMap<int>> mvaPUIDToken_;

  edm::EDGetTokenT<edm::ValueMap<int>> qgMultiplicityToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> qgLikelihoodToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> qgptDToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> qgaxis2Token_;

  //edm::EDGetTokenT<reco::JPTJetCollection>        jptJetsToken_;

  edm::InputTag inputJetIDValueMap;
  edm::EDGetTokenT<edm::ValueMap<reco::JetID>> jetID_ValueMapToken_;
  edm::ESGetToken<L1GtTriggerMenu, L1GtTriggerMenuRcd> l1gtTrigMenuToken_;

  //Cleaning parameters
  edm::ParameterSet cleaningParameters_;
  edm::InputTag vertexLabel_;
  edm::InputTag gtLabel_;

  //check later if we need only one set of parameters
  edm::ParameterSet parameters_;

  edm::InputTag jetCorrectorTag_;
  edm::EDGetTokenT<reco::JetCorrector> jetCorrectorToken_;

  std::string JetIDQuality_;
  std::string JetIDVersion_;
  JetIDSelectionFunctor::Quality_t jetidquality;
  JetIDSelectionFunctor::Version_t jetidversion;
  JetIDSelectionFunctor jetIDFunctor;

  PFJetIDSelectionFunctor::Quality_t pfjetidquality;
  PFJetIDSelectionFunctor::Version_t pfjetidversion;

  PFJetIDSelectionFunctor pfjetIDFunctor;

  std::vector<std::string> folderNames_;

  std::string DirName;

  // Book MonitorElements
  void bookMESetSelection(std::string, DQMStore::IBooker&);
  //void bookMonitorElement(std::string, bool);

  int verbose_;
  //histo binning parameters -> these are PART of ALL analyzers - move it up
  int etaBin_;
  double etaMin_;
  double etaMax_;

  int phiBin_;
  double phiMin_;
  double phiMax_;

  int ptBin_;
  double ptMin_;
  double ptMax_;

  int eBin_;
  double eMin_;
  double eMax_;

  int pBin_;
  double pMin_;
  double pMax_;

  int nbinsPV_;
  double nPVlow_;
  double nPVhigh_;

  //variables which are present both in
  int jetLoPass_;
  int jetHiPass_;
  int leadJetFlag_;
  double ptThreshold_;
  double ptThresholdUnc_;
  double asymmetryThirdJetCut_;
  double balanceThirdJetCut_;

  //
  int fillJIDPassFrac_;
  std::string m_l1algoname_;
  int m_bitAlgTechTrig_;

  //the histos
  MonitorElement* jetME;

  // --- Used for Data Certification - use for Calo, PF and JPT jets
  MonitorElement* mPt;
  MonitorElement* mPt_1;
  MonitorElement* mPt_2;
  MonitorElement* mPt_3;
  MonitorElement* mEta;
  MonitorElement* mPhi;
  MonitorElement* mPt_uncor;
  MonitorElement* mEta_uncor;
  MonitorElement* mPhi_uncor;
  MonitorElement* mConstituents_uncor;

  MonitorElement* mJetEnergyCorr;
  MonitorElement* mJetEnergyCorrVSEta;
  MonitorElement* mJetEnergyCorrVSPt;

  MonitorElement* mConstituents;
  MonitorElement* mHFrac;
  MonitorElement* mEFrac;
  MonitorElement* mPhiVSEta;

  MonitorElement* mPt_Barrel;
  MonitorElement* mPhi_Barrel;
  MonitorElement* mConstituents_Barrel;
  MonitorElement* mHFrac_Barrel;
  MonitorElement* mEFrac_Barrel;

  MonitorElement* mPt_EndCap;
  MonitorElement* mPhi_EndCap;
  MonitorElement* mConstituents_EndCap;
  MonitorElement* mHFrac_EndCap;
  MonitorElement* mEFrac_EndCap;

  MonitorElement* mPt_Forward;
  MonitorElement* mPhi_Forward;
  MonitorElement* mConstituents_Forward;
  MonitorElement* mHFrac_Forward;
  MonitorElement* mEFrac_Forward;

  MonitorElement* mPt_Barrel_Hi;
  MonitorElement* mPhi_Barrel_Hi;
  MonitorElement* mConstituents_Barrel_Hi;
  MonitorElement* mHFrac_Barrel_Hi;

  MonitorElement* mPt_EndCap_Hi;
  MonitorElement* mPhi_EndCap_Hi;
  MonitorElement* mConstituents_EndCap_Hi;
  MonitorElement* mHFrac_EndCap_Hi;

  MonitorElement* mPt_Forward_Hi;
  MonitorElement* mPhi_Forward_Hi;
  MonitorElement* mConstituents_Forward_Hi;
  MonitorElement* mHFrac_Forward_Hi;

  MonitorElement* mNJets;
  MonitorElement* mDPhi;

  // Leading Jet Parameters
  MonitorElement* mEtaFirst;
  MonitorElement* mPhiFirst;
  MonitorElement* mPtFirst;

  // Events passing the jet triggers
  MonitorElement* mPhi_Lo;
  MonitorElement* mPt_Lo;

  MonitorElement* mEta_Hi;
  MonitorElement* mPhi_Hi;
  MonitorElement* mPt_Hi;

  MonitorElement* mLooseJIDPassFractionVSeta;
  MonitorElement* mLooseJIDPassFractionVSpt;
  MonitorElement* mLooseJIDPassFractionVSptNoHF;

  MonitorElement* mLooseMVAPUJIDPassFractionVSeta;
  MonitorElement* mLooseMVAPUJIDPassFractionVSpt;
  MonitorElement* mMediumMVAPUJIDPassFractionVSeta;
  MonitorElement* mMediumMVAPUJIDPassFractionVSpt;
  MonitorElement* mTightMVAPUJIDPassFractionVSeta;
  MonitorElement* mTightMVAPUJIDPassFractionVSpt;
  MonitorElement* mMVAPUJIDDiscriminant_lowPt_Barrel;
  MonitorElement* mMVAPUJIDDiscriminant_lowPt_EndCap;
  MonitorElement* mMVAPUJIDDiscriminant_lowPt_Forward;
  MonitorElement* mMVAPUJIDDiscriminant_mediumPt_Barrel;
  MonitorElement* mMVAPUJIDDiscriminant_mediumPt_EndCap;
  MonitorElement* mMVAPUJIDDiscriminant_mediumPt_Forward;
  MonitorElement* mMVAPUJIDDiscriminant_highPt_Barrel;
  MonitorElement* mMVAPUJIDDiscriminant_highPt_EndCap;
  MonitorElement* mMVAPUJIDDiscriminant_highPt_Forward;

  MonitorElement* mLooseCutPUJIDPassFractionVSeta;
  MonitorElement* mLooseCutPUJIDPassFractionVSpt;
  MonitorElement* mMediumCutPUJIDPassFractionVSeta;
  MonitorElement* mMediumCutPUJIDPassFractionVSpt;
  MonitorElement* mTightCutPUJIDPassFractionVSeta;
  MonitorElement* mTightCutPUJIDPassFractionVSpt;
  MonitorElement* mCutPUJIDDiscriminant_lowPt_Barrel;
  MonitorElement* mCutPUJIDDiscriminant_lowPt_EndCap;
  MonitorElement* mCutPUJIDDiscriminant_lowPt_Forward;
  MonitorElement* mCutPUJIDDiscriminant_mediumPt_Barrel;
  MonitorElement* mCutPUJIDDiscriminant_mediumPt_EndCap;
  MonitorElement* mCutPUJIDDiscriminant_mediumPt_Forward;
  MonitorElement* mCutPUJIDDiscriminant_highPt_Barrel;
  MonitorElement* mCutPUJIDDiscriminant_highPt_EndCap;
  MonitorElement* mCutPUJIDDiscriminant_highPt_Forward;

  //dijet analysis quantities
  MonitorElement* mDijetBalance;
  MonitorElement* mDijetAsymmetry;

  // NPV profiles
  //----------------------------------------------------------------------------
  MonitorElement* mNJets_profile;
  MonitorElement* mPt_profile;
  MonitorElement* mEta_profile;
  MonitorElement* mPhi_profile;
  MonitorElement* mConstituents_profile;
  MonitorElement* mHFrac_profile;
  MonitorElement* mEFrac_profile;

  bool hltInitialized_;
  bool bypassAllPVChecks_;

  HLTConfigProvider hltConfig_;
  std::string processname_;

  //MonitorElement* hltpathME;
  MonitorElement* cleanupME;
  MonitorElement* verticesME;

  GenericTriggerEventFlag* highPtJetEventFlag_;
  GenericTriggerEventFlag* lowPtJetEventFlag_;

  std::vector<std::string> highPtJetExpr_;
  std::vector<std::string> lowPtJetExpr_;

  bool jetCleaningFlag_;
  bool filljetsubstruc_;
  double pt_min_boosted_;

  bool runcosmics_;

  //  bool energycorrected;

  // CaloJet specific
  MonitorElement* mMaxEInEmTowers;
  MonitorElement* mMaxEInHadTowers;
  MonitorElement* mHadEnergyInHO;
  MonitorElement* mHadEnergyInHB;
  MonitorElement* mHadEnergyInHF;
  MonitorElement* mHadEnergyInHE;
  MonitorElement* mEmEnergyInEB;
  MonitorElement* mEmEnergyInEE;
  MonitorElement* mEmEnergyInHF;
  MonitorElement* mN90Hits;
  MonitorElement* mfHPD;
  MonitorElement* mfRBX;
  MonitorElement* mresEMF;
  MonitorElement* mEMF;

  //now define PFJet only flags
  MonitorElement* mCHFrac_lowPt_Barrel;
  MonitorElement* mNHFrac_lowPt_Barrel;
  MonitorElement* mPhFrac_lowPt_Barrel;
  MonitorElement* mCHFrac_mediumPt_Barrel;
  MonitorElement* mNHFrac_mediumPt_Barrel;
  MonitorElement* mPhFrac_mediumPt_Barrel;
  MonitorElement* mCHFrac_highPt_Barrel;
  MonitorElement* mNHFrac_highPt_Barrel;
  MonitorElement* mPhFrac_highPt_Barrel;
  MonitorElement* mCHEn_lowPt_Barrel;
  MonitorElement* mNHEn_lowPt_Barrel;
  MonitorElement* mPhEn_lowPt_Barrel;
  MonitorElement* mElEn_lowPt_Barrel;
  MonitorElement* mMuEn_lowPt_Barrel;
  MonitorElement* mCHEn_mediumPt_Barrel;
  MonitorElement* mNHEn_mediumPt_Barrel;
  MonitorElement* mPhEn_mediumPt_Barrel;
  MonitorElement* mElEn_mediumPt_Barrel;
  MonitorElement* mMuEn_mediumPt_Barrel;
  MonitorElement* mCHEn_highPt_Barrel;
  MonitorElement* mNHEn_highPt_Barrel;
  MonitorElement* mPhEn_highPt_Barrel;
  MonitorElement* mElEn_highPt_Barrel;
  MonitorElement* mMuEn_highPt_Barrel;
  MonitorElement* mChMultiplicity_lowPt_Barrel;
  MonitorElement* mNeutMultiplicity_lowPt_Barrel;
  MonitorElement* mMuMultiplicity_lowPt_Barrel;
  MonitorElement* mChMultiplicity_mediumPt_Barrel;
  MonitorElement* mNeutMultiplicity_mediumPt_Barrel;
  MonitorElement* mMuMultiplicity_mediumPt_Barrel;
  MonitorElement* mChMultiplicity_highPt_Barrel;
  MonitorElement* mNeutMultiplicity_highPt_Barrel;
  MonitorElement* mMuMultiplicity_highPt_Barrel;

  MonitorElement* mCHFracVSpT_Barrel;
  MonitorElement* mNHFracVSpT_Barrel;
  MonitorElement* mPhFracVSpT_Barrel;
  MonitorElement* mCHFracVSpT_EndCap;
  MonitorElement* mNHFracVSpT_EndCap;
  MonitorElement* mPhFracVSpT_EndCap;
  MonitorElement* mHFHFracVSpT_Forward;
  MonitorElement* mHFEFracVSpT_Forward;

  MonitorElement* mCHFracVSeta_lowPt;
  MonitorElement* mNHFracVSeta_lowPt;
  MonitorElement* mPhFracVSeta_lowPt;
  MonitorElement* mCHFracVSeta_mediumPt;
  MonitorElement* mNHFracVSeta_mediumPt;
  MonitorElement* mPhFracVSeta_mediumPt;
  MonitorElement* mCHFracVSeta_highPt;
  MonitorElement* mNHFracVSeta_highPt;
  MonitorElement* mPhFracVSeta_highPt;

  MonitorElement* mCHFrac_lowPt_EndCap;
  MonitorElement* mNHFrac_lowPt_EndCap;
  MonitorElement* mPhFrac_lowPt_EndCap;
  MonitorElement* mCHFrac_mediumPt_EndCap;
  MonitorElement* mNHFrac_mediumPt_EndCap;
  MonitorElement* mPhFrac_mediumPt_EndCap;
  MonitorElement* mCHFrac_highPt_EndCap;
  MonitorElement* mNHFrac_highPt_EndCap;
  MonitorElement* mPhFrac_highPt_EndCap;

  MonitorElement* mCHEn_lowPt_EndCap;
  MonitorElement* mNHEn_lowPt_EndCap;
  MonitorElement* mPhEn_lowPt_EndCap;
  MonitorElement* mElEn_lowPt_EndCap;
  MonitorElement* mMuEn_lowPt_EndCap;
  MonitorElement* mCHEn_mediumPt_EndCap;
  MonitorElement* mNHEn_mediumPt_EndCap;
  MonitorElement* mPhEn_mediumPt_EndCap;
  MonitorElement* mElEn_mediumPt_EndCap;
  MonitorElement* mMuEn_mediumPt_EndCap;
  MonitorElement* mCHEn_highPt_EndCap;
  MonitorElement* mNHEn_highPt_EndCap;
  MonitorElement* mPhEn_highPt_EndCap;
  MonitorElement* mElEn_highPt_EndCap;
  MonitorElement* mMuEn_highPt_EndCap;
  MonitorElement* mMass_lowPt_Barrel;
  MonitorElement* mMass_lowPt_EndCap;
  MonitorElement* mMass_lowPt_Forward;
  MonitorElement* mMass_mediumPt_Barrel;
  MonitorElement* mMass_mediumPt_EndCap;
  MonitorElement* mMass_mediumPt_Forward;
  MonitorElement* mMass_highPt_Barrel;
  MonitorElement* mMass_highPt_EndCap;
  MonitorElement* mMass_highPt_Forward;

  MonitorElement* mChMultiplicity_lowPt_EndCap;
  MonitorElement* mNeutMultiplicity_lowPt_EndCap;
  MonitorElement* mMuMultiplicity_lowPt_EndCap;
  MonitorElement* mChMultiplicity_mediumPt_EndCap;
  MonitorElement* mNeutMultiplicity_mediumPt_EndCap;
  MonitorElement* mMuMultiplicity_mediumPt_EndCap;
  MonitorElement* mChMultiplicity_highPt_EndCap;
  MonitorElement* mNeutMultiplicity_highPt_EndCap;
  MonitorElement* mMuMultiplicity_highPt_EndCap;

  MonitorElement* mHFEFrac_lowPt_Forward;
  MonitorElement* mHFHFrac_lowPt_Forward;
  MonitorElement* mHFEFrac_mediumPt_Forward;
  MonitorElement* mHFHFrac_mediumPt_Forward;
  MonitorElement* mHFEFrac_highPt_Forward;
  MonitorElement* mHFHFrac_highPt_Forward;
  MonitorElement* mHFEEn_lowPt_Forward;
  MonitorElement* mHFHEn_lowPt_Forward;
  MonitorElement* mHFEEn_mediumPt_Forward;
  MonitorElement* mHFHEn_mediumPt_Forward;
  MonitorElement* mHFEEn_highPt_Forward;
  MonitorElement* mHFHEn_highPt_Forward;
  MonitorElement* mNeutMultiplicity_lowPt_Forward;
  MonitorElement* mNeutMultiplicity_mediumPt_Forward;
  MonitorElement* mNeutMultiplicity_highPt_Forward;

  MonitorElement* mChargedHadronEnergy;
  MonitorElement* mNeutralHadronEnergy;
  MonitorElement* mChargedEmEnergy;
  MonitorElement* mChargedMuEnergy;
  MonitorElement* mNeutralEmEnergy;
  MonitorElement* mChargedMultiplicity;
  MonitorElement* mNeutralMultiplicity;
  MonitorElement* mMuonMultiplicity;

  //it is there for ak4PFCHS
  MonitorElement* mAxis2_lowPt_Barrel;
  MonitorElement* mpTD_lowPt_Barrel;
  MonitorElement* mMultiplicityQG_lowPt_Barrel;
  MonitorElement* mqgLikelihood_lowPt_Barrel;
  MonitorElement* mAxis2_mediumPt_Barrel;
  MonitorElement* mpTD_mediumPt_Barrel;
  MonitorElement* mMultiplicityQG_mediumPt_Barrel;
  MonitorElement* mqgLikelihood_mediumPt_Barrel;
  MonitorElement* mAxis2_highPt_Barrel;
  MonitorElement* mpTD_highPt_Barrel;
  MonitorElement* mMultiplicityQG_highPt_Barrel;
  MonitorElement* mqgLikelihood_highPt_Barrel;

  MonitorElement* mAxis2_lowPt_EndCap;
  MonitorElement* mpTD_lowPt_EndCap;
  MonitorElement* mMultiplicityQG_lowPt_EndCap;
  MonitorElement* mqgLikelihood_lowPt_EndCap;
  MonitorElement* mAxis2_mediumPt_EndCap;
  MonitorElement* mpTD_mediumPt_EndCap;
  MonitorElement* mMultiplicityQG_mediumPt_EndCap;
  MonitorElement* mqgLikelihood_mediumPt_EndCap;
  MonitorElement* mAxis2_highPt_EndCap;
  MonitorElement* mpTD_highPt_EndCap;
  MonitorElement* mMultiplicityQG_highPt_EndCap;
  MonitorElement* mqgLikelihood_highPt_EndCap;

  MonitorElement* mAxis2_lowPt_Forward;
  MonitorElement* mpTD_lowPt_Forward;
  MonitorElement* mMultiplicityQG_lowPt_Forward;
  MonitorElement* mqgLikelihood_lowPt_Forward;
  MonitorElement* mAxis2_mediumPt_Forward;
  MonitorElement* mpTD_mediumPt_Forward;
  MonitorElement* mMultiplicityQG_mediumPt_Forward;
  MonitorElement* mqgLikelihood_mediumPt_Forward;
  MonitorElement* mAxis2_highPt_Forward;
  MonitorElement* mpTD_highPt_Forward;
  MonitorElement* mMultiplicityQG_highPt_Forward;
  MonitorElement* mqgLikelihood_highPt_Forward;

  //new Plots with Res./ Eff. as function of neutral, charged &  em fraction

  MonitorElement* mNeutralFraction;
  MonitorElement* mNeutralFraction2;

  MonitorElement* mEEffNeutralFraction;
  MonitorElement* mEEffChargedFraction;
  MonitorElement* mEResNeutralFraction;
  MonitorElement* mEResChargedFraction;
  MonitorElement* nEEff;
  //PF specific NPV profiles
  MonitorElement* mChargedHadronEnergy_profile;
  MonitorElement* mNeutralHadronEnergy_profile;
  MonitorElement* mChargedEmEnergy_profile;
  MonitorElement* mChargedMuEnergy_profile;
  MonitorElement* mNeutralEmEnergy_profile;
  MonitorElement* mChargedMultiplicity_profile;
  MonitorElement* mNeutralMultiplicity_profile;
  MonitorElement* mMuonMultiplicity_profile;

  //Monitor Elements for special selections
  //for special selections
  MonitorElement* mCHFrac;
  MonitorElement* mNHFrac;
  MonitorElement* mPhFrac;
  MonitorElement* mHFEMFrac;
  MonitorElement* mHFHFrac;
  MonitorElement* mCHFrac_profile;
  MonitorElement* mNHFrac_profile;
  MonitorElement* mPhFrac_profile;
  MonitorElement* mHFEMFrac_profile;
  MonitorElement* mHFHFrac_profile;

  JetMETDQMDCSFilter* DCSFilterForJetMonitoring_;
  JetMETDQMDCSFilter* DCSFilterForDCSMonitoring_;
  /*
  MonitorElement* mePhFracBarrel_BXm2BXm1Empty;
  MonitorElement* meNHFracBarrel_BXm2BXm1Empty;
  MonitorElement* meCHFracBarrel_BXm2BXm1Empty;
  MonitorElement* mePtBarrel_BXm2BXm1Empty;
  MonitorElement* mePhFracEndCapMinus_BXm2BXm1Empty;
  MonitorElement* meNHFracEndCapMinus_BXm2BXm1Empty;
  MonitorElement* meCHFracEndCapMinus_BXm2BXm1Empty;
  MonitorElement* mePtEndCapMinus_BXm2BXm1Empty;
  MonitorElement* mePhFracEndCapPlus_BXm2BXm1Empty;
  MonitorElement* meNHFracEndCapPlus_BXm2BXm1Empty;
  MonitorElement* meCHFracEndCapPlus_BXm2BXm1Empty;
  MonitorElement* mePtEndCapPlus_BXm2BXm1Empty;
  MonitorElement* meHFHFracMinus_BXm2BXm1Empty;
  MonitorElement* meHFEMFracMinus_BXm2BXm1Empty;
  MonitorElement* mePtForwardMinus_BXm2BXm1Empty;
  MonitorElement* meHFHFracPlus_BXm2BXm1Empty;
  MonitorElement* meHFEMFracPlus_BXm2BXm1Empty;
  MonitorElement* mePtForwardPlus_BXm2BXm1Empty;
  MonitorElement* meEta_BXm2BXm1Empty;
  */
  MonitorElement* mePhFracBarrel_BXm1Empty;
  MonitorElement* meNHFracBarrel_BXm1Empty;
  MonitorElement* meCHFracBarrel_BXm1Empty;
  MonitorElement* mePtBarrel_BXm1Empty;
  MonitorElement* mePhFracEndCapMinus_BXm1Empty;
  MonitorElement* meNHFracEndCapMinus_BXm1Empty;
  MonitorElement* meCHFracEndCapMinus_BXm1Empty;
  MonitorElement* mePtEndCapMinus_BXm1Empty;
  MonitorElement* mePhFracEndCapPlus_BXm1Empty;
  MonitorElement* meNHFracEndCapPlus_BXm1Empty;
  MonitorElement* meCHFracEndCapPlus_BXm1Empty;
  MonitorElement* mePtEndCapPlus_BXm1Empty;
  MonitorElement* meHFHFracMinus_BXm1Empty;
  MonitorElement* meHFEMFracMinus_BXm1Empty;
  MonitorElement* mePtForwardMinus_BXm1Empty;
  MonitorElement* meHFHFracPlus_BXm1Empty;
  MonitorElement* meHFEMFracPlus_BXm1Empty;
  MonitorElement* mePtForwardPlus_BXm1Empty;
  MonitorElement* meEta_BXm1Empty;
  /*
  MonitorElement* mePhFracBarrel_BXm2BXm1Filled;
  MonitorElement* meNHFracBarrel_BXm2BXm1Filled;
  MonitorElement* meCHFracBarrel_BXm2BXm1Filled;
  MonitorElement* mePtBarrel_BXm2BXm1Filled;
  MonitorElement* mePhFracEndCapMinus_BXm2BXm1Filled;
  MonitorElement* meNHFracEndCapMinus_BXm2BXm1Filled;
  MonitorElement* meCHFracEndCapMinus_BXm2BXm1Filled;
  MonitorElement* mePtEndCapMinus_BXm2BXm1Filled;
  MonitorElement* mePhFracEndCapPlus_BXm2BXm1Filled;
  MonitorElement* meNHFracEndCapPlus_BXm2BXm1Filled;
  MonitorElement* meCHFracEndCapPlus_BXm2BXm1Filled;
  MonitorElement* mePtEndCapPlus_BXm2BXm1Filled;
  MonitorElement* meHFHFracMinus_BXm2BXm1Filled;
  MonitorElement* meHFEMFracMinus_BXm2BXm1Filled;
  MonitorElement* mePtForwardMinus_BXm2BXm1Filled;
  MonitorElement* meHFHFracPlus_BXm2BXm1Filled;
  MonitorElement* meHFEMFracPlus_BXm2BXm1Filled;
  MonitorElement* mePtForwardPlus_BXm2BXm1Filled;
  MonitorElement* meEta_BXm2BXm1Filled;
  */
  MonitorElement* mePhFracBarrel_BXm1Filled;
  MonitorElement* meNHFracBarrel_BXm1Filled;
  MonitorElement* meCHFracBarrel_BXm1Filled;
  MonitorElement* mePtBarrel_BXm1Filled;
  MonitorElement* mePhFracEndCapMinus_BXm1Filled;
  MonitorElement* meNHFracEndCapMinus_BXm1Filled;
  MonitorElement* meCHFracEndCapMinus_BXm1Filled;
  MonitorElement* mePtEndCapMinus_BXm1Filled;
  MonitorElement* mePhFracEndCapPlus_BXm1Filled;
  MonitorElement* meNHFracEndCapPlus_BXm1Filled;
  MonitorElement* meCHFracEndCapPlus_BXm1Filled;
  MonitorElement* mePtEndCapPlus_BXm1Filled;
  MonitorElement* meHFHFracMinus_BXm1Filled;
  MonitorElement* meHFEMFracMinus_BXm1Filled;
  MonitorElement* mePtForwardMinus_BXm1Filled;
  MonitorElement* meHFHFracPlus_BXm1Filled;
  MonitorElement* meHFEMFracPlus_BXm1Filled;
  MonitorElement* mePtForwardPlus_BXm1Filled;
  MonitorElement* meEta_BXm1Filled;

  //miniaod specific variables, especially for substructure
  MonitorElement* mSoftDropMass;
  MonitorElement* mPrunedMass;
  MonitorElement* mTrimmedMass;
  MonitorElement* mFilteredMass;
  MonitorElement* mtau2_over_tau1;
  MonitorElement* mtau3_over_tau2;
  MonitorElement* mCATopTag_topMass;
  MonitorElement* mCATopTag_minMass;
  MonitorElement* mCATopTag_nSubJets;

  MonitorElement* mnSubJetsCMSTopTag;
  MonitorElement* mSubJet1_CMSTopTag_pt;
  MonitorElement* mSubJet1_CMSTopTag_eta;
  MonitorElement* mSubJet1_CMSTopTag_phi;
  MonitorElement* mSubJet1_CMSTopTag_mass;
  MonitorElement* mSubJet2_CMSTopTag_pt;
  MonitorElement* mSubJet2_CMSTopTag_eta;
  MonitorElement* mSubJet2_CMSTopTag_phi;
  MonitorElement* mSubJet2_CMSTopTag_mass;
  MonitorElement* mSubJet3_CMSTopTag_pt;
  MonitorElement* mSubJet3_CMSTopTag_eta;
  MonitorElement* mSubJet3_CMSTopTag_phi;
  MonitorElement* mSubJet3_CMSTopTag_mass;
  MonitorElement* mSubJet4_CMSTopTag_pt;
  MonitorElement* mSubJet4_CMSTopTag_eta;
  MonitorElement* mSubJet4_CMSTopTag_phi;
  MonitorElement* mSubJet4_CMSTopTag_mass;

  MonitorElement* mnSubJetsSoftDrop;
  MonitorElement* mSubJet1_SoftDrop_pt;
  MonitorElement* mSubJet1_SoftDrop_eta;
  MonitorElement* mSubJet1_SoftDrop_phi;
  MonitorElement* mSubJet1_SoftDrop_mass;
  MonitorElement* mSubJet2_SoftDrop_pt;
  MonitorElement* mSubJet2_SoftDrop_eta;
  MonitorElement* mSubJet2_SoftDrop_phi;
  MonitorElement* mSubJet2_SoftDrop_mass;

  //miniaod specific variables, especially for substructure for a boosted regime
  MonitorElement* mSoftDropMass_boosted;
  MonitorElement* mPrunedMass_boosted;
  MonitorElement* mTrimmedMass_boosted;
  MonitorElement* mFilteredMass_boosted;
  MonitorElement* mtau2_over_tau1_boosted;
  MonitorElement* mtau3_over_tau2_boosted;
  MonitorElement* mCATopTag_topMass_boosted;
  MonitorElement* mCATopTag_minMass_boosted;
  MonitorElement* mCATopTag_nSubJets_boosted;

  MonitorElement* mnSubJetsCMSTopTag_boosted;
  MonitorElement* mSubJet1_CMSTopTag_pt_boosted;
  MonitorElement* mSubJet1_CMSTopTag_eta_boosted;
  MonitorElement* mSubJet1_CMSTopTag_phi_boosted;
  MonitorElement* mSubJet1_CMSTopTag_mass_boosted;
  MonitorElement* mSubJet2_CMSTopTag_pt_boosted;
  MonitorElement* mSubJet2_CMSTopTag_eta_boosted;
  MonitorElement* mSubJet2_CMSTopTag_phi_boosted;
  MonitorElement* mSubJet2_CMSTopTag_mass_boosted;
  MonitorElement* mSubJet3_CMSTopTag_pt_boosted;
  MonitorElement* mSubJet3_CMSTopTag_eta_boosted;
  MonitorElement* mSubJet3_CMSTopTag_phi_boosted;
  MonitorElement* mSubJet3_CMSTopTag_mass_boosted;
  MonitorElement* mSubJet4_CMSTopTag_pt_boosted;
  MonitorElement* mSubJet4_CMSTopTag_eta_boosted;
  MonitorElement* mSubJet4_CMSTopTag_phi_boosted;
  MonitorElement* mSubJet4_CMSTopTag_mass_boosted;

  MonitorElement* mnSubJetsSoftDrop_boosted;
  MonitorElement* mSubJet1_SoftDrop_pt_boosted;
  MonitorElement* mSubJet1_SoftDrop_eta_boosted;
  MonitorElement* mSubJet1_SoftDrop_phi_boosted;
  MonitorElement* mSubJet1_SoftDrop_mass_boosted;
  MonitorElement* mSubJet2_SoftDrop_pt_boosted;
  MonitorElement* mSubJet2_SoftDrop_eta_boosted;
  MonitorElement* mSubJet2_SoftDrop_phi_boosted;
  MonitorElement* mSubJet2_SoftDrop_mass_boosted;

  //miniaod only variables
  MonitorElement* mPt_CaloJet;
  MonitorElement* mEMF_CaloJet;
  MonitorElement* mMass_Barrel;
  MonitorElement* mMass_EndCap;
  MonitorElement* mMass_Forward;

  //now ZJets plots
  MonitorElement* mDPhiZJet;
  MonitorElement* mZMass;
  MonitorElement* mZJetAsymmetry;
  MonitorElement* mJetZBalance_lowZPt_J_Barrel;
  MonitorElement* mJetZBalance_mediumZPt_J_Barrel;
  MonitorElement* mJetZBalance_highZPt_J_Barrel;
  MonitorElement* mJetZBalance_lowZPt_J_EndCap;
  MonitorElement* mJetZBalance_mediumZPt_J_EndCap;
  MonitorElement* mJetZBalance_highZPt_J_EndCap;
  MonitorElement* mJetZBalance_lowZPt_J_Forward;
  MonitorElement* mJetZBalance_mediumZPt_J_Forward;
  MonitorElement* mJetZBalance_highZPt_J_Forward;
  MonitorElement* mJ1Pt_over_ZPt_J_Barrel;
  MonitorElement* mJ1Pt_over_ZPt_J_EndCap;
  MonitorElement* mJ1Pt_over_ZPt_J_Forward;
  MonitorElement* mMPF_J_Barrel;
  MonitorElement* mMPF_J_EndCap;
  MonitorElement* mMPF_J_Forward;
  MonitorElement* mJ1Pt_over_ZPt_lowZPt_J_Barrel;
  MonitorElement* mJ1Pt_over_ZPt_mediumZPt_J_Barrel;
  MonitorElement* mJ1Pt_over_ZPt_highZPt_J_Barrel;
  MonitorElement* mJ1Pt_over_ZPt_lowZPt_J_EndCap;
  MonitorElement* mJ1Pt_over_ZPt_mediumZPt_J_EndCap;
  MonitorElement* mJ1Pt_over_ZPt_highZPt_J_EndCap;
  MonitorElement* mJ1Pt_over_ZPt_lowZPt_J_Forward;
  MonitorElement* mJ1Pt_over_ZPt_mediumZPt_J_Forward;
  MonitorElement* mJ1Pt_over_ZPt_highZPt_J_Forward;
  MonitorElement* mMPF_lowZPt_J_Barrel;
  MonitorElement* mMPF_mediumZPt_J_Barrel;
  MonitorElement* mMPF_highZPt_J_Barrel;
  MonitorElement* mMPF_lowZPt_J_EndCap;
  MonitorElement* mMPF_mediumZPt_J_EndCap;
  MonitorElement* mMPF_highZPt_J_EndCap;
  MonitorElement* mMPF_lowZPt_J_Forward;
  MonitorElement* mMPF_mediumZPt_J_Forward;
  MonitorElement* mMPF_highZPt_J_Forward;
  MonitorElement* mDeltaPt_Z_j1_over_ZPt_30_55_J_Barrel;
  MonitorElement* mDeltaPt_Z_j1_over_ZPt_55_75_J_Barrel;
  MonitorElement* mDeltaPt_Z_j1_over_ZPt_75_150_J_Barrel;
  MonitorElement* mDeltaPt_Z_j1_over_ZPt_150_290_J_Barrel;
  MonitorElement* mDeltaPt_Z_j1_over_ZPt_290_J_Barrel;
  MonitorElement* mDeltaPt_Z_j1_over_ZPt_30_55_J_EndCap;
  MonitorElement* mDeltaPt_Z_j1_over_ZPt_55_75_J_EndCap;
  MonitorElement* mDeltaPt_Z_j1_over_ZPt_75_150_J_EndCap;
  MonitorElement* mDeltaPt_Z_j1_over_ZPt_150_290_J_EndCap;
  MonitorElement* mDeltaPt_Z_j1_over_ZPt_290_J_EndCap;
  MonitorElement* mDeltaPt_Z_j1_over_ZPt_30_55_J_Forward;
  MonitorElement* mDeltaPt_Z_j1_over_ZPt_55_100_J_Forward;
  MonitorElement* mDeltaPt_Z_j1_over_ZPt_100_J_Forward;

  std::map<std::string, MonitorElement*> map_of_MEs;

  bool isCaloJet_;
  bool isPFJet_;
  bool isMiniAODJet_;

  bool fill_jet_high_level_histo;

  bool fill_CHS_histos;
};
#endif
