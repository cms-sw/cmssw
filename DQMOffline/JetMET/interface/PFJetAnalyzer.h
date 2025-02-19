#ifndef PFJetAnalyzer_H
#define PFJetAnalyzer_H


/** \class PFJetAnalyzer
 *
 *  DQM monitoring source for PFlow Jets
 *
 *  $Date: 2012/03/23 18:24:43 $
 *  $Revision: 1.11 $
 *  \author F. Chlebana - Fermilab
 */


#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMOffline/JetMET/interface/PFJetAnalyzerBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
//#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"


#include "GlobalVariables.h"


class PFJetAnalyzer : public PFJetAnalyzerBase {
 public:

  /// Constructor
  PFJetAnalyzer(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~PFJetAnalyzer();

  /// Inizialize parameters for histo binning
  void beginJob(DQMStore * dbe);

  /// Finish up a job
  void endJob();

  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&, const reco::PFJetCollection& pfJets, const int numPV);
  //
  void setSource(std::string source) {
    _source = source;
  }
  //
  void setLeadJetFlag(int flag) {
    _leadJetFlag = flag;
  }
  int getLeadJetFlag() {
    return  _leadJetFlag;
  }
  void setJetLoPass(int pass) {
    _JetLoPass = pass;
  }

  void setJetHiPass(int pass) {
    _JetHiPass = pass;
  }


 private:
  // ----------member data ---------------------------
  int   _JetLoPass;
  int   _JetHiPass;
  int   _leadJetFlag;
  double _ptThreshold;

  double _asymmetryThirdJetCut;
  double _balanceThirdJetCut;

  int makedijetselection;

  //histo binning parameters
  int    etaBin;
  double etaMin;
  double etaMax;

  int    phiBin;
  double phiMin;
  double phiMax;

  int    ptBin;
  double ptMin;
  double ptMax;

  int    eBin;
  double eMin;
  double eMax;

  int    pBin;
  double pMin;
  double pMax;

  int fillpfJIDPassFrac;

  double _ThisCHFMin;
  double _ThisNHFMax;
  double _ThisCEFMax;
  double _ThisNEFMax;
  double _LooseCHFMin;
  double _LooseNHFMax;
  double _LooseCEFMax;
  double _LooseNEFMax;
  double _TightCHFMin;
  double _TightNHFMax;
  double _TightCEFMax;
  double _TightNEFMax;

  //the histos
  MonitorElement* jetME;

  edm::ParameterSet parameters;
  // Switch for verbosity
  std::string metname;
  // Calo Jet Label
  edm::InputTag thePFJetCollectionLabel;
  std::string _source;

/*   // Generic Jet Parameters */
/*   MonitorElement* mEta; */
/*   MonitorElement* mPhi; */
/*   MonitorElement* mE; */
/*   MonitorElement* mP; */
/*   MonitorElement* mPt; */
/*   MonitorElement* mMass; */
/*   MonitorElement* mConstituents; */

/*   // Leading Jet Parameters */
/*   MonitorElement* mEtaFirst; */
/*   MonitorElement* mPhiFirst; */
/*   MonitorElement* mEFirst; */
/*   MonitorElement* mPtFirst; */
  MonitorElement* mPt;
  MonitorElement* mPt_1;
  MonitorElement* mPt_2;
  MonitorElement* mPt_3;
  MonitorElement* mEta;
  MonitorElement* mPhi;
  MonitorElement* mConstituents;
  MonitorElement* mHFrac;
  MonitorElement* mEFrac;
  MonitorElement* mPhiVSEta;

  MonitorElement* mPt_Barrel;
  MonitorElement* mPhi_Barrel;

  MonitorElement* mCHFrac_lowPt_Barrel;
  MonitorElement* mNHFrac_lowPt_Barrel;
  MonitorElement* mPhFrac_lowPt_Barrel;
  MonitorElement* mElFrac_lowPt_Barrel;
  MonitorElement* mMuFrac_lowPt_Barrel;
  MonitorElement* mCHFrac_mediumPt_Barrel;
  MonitorElement* mNHFrac_mediumPt_Barrel;
  MonitorElement* mPhFrac_mediumPt_Barrel;
  MonitorElement* mElFrac_mediumPt_Barrel;
  MonitorElement* mMuFrac_mediumPt_Barrel;
  MonitorElement* mCHFrac_highPt_Barrel;
  MonitorElement* mNHFrac_highPt_Barrel;
  MonitorElement* mPhFrac_highPt_Barrel;
  MonitorElement* mElFrac_highPt_Barrel;
  MonitorElement* mMuFrac_highPt_Barrel;
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
  MonitorElement* mNeuMultiplicity_lowPt_Barrel;
  MonitorElement* mMuMultiplicity_lowPt_Barrel;
  MonitorElement* mChMultiplicity_mediumPt_Barrel;
  MonitorElement* mNeuMultiplicity_mediumPt_Barrel;
  MonitorElement* mMuMultiplicity_mediumPt_Barrel;
  MonitorElement* mChMultiplicity_highPt_Barrel;
  MonitorElement* mNeuMultiplicity_highPt_Barrel;
  MonitorElement* mMuMultiplicity_highPt_Barrel;

  MonitorElement*  mCHFracVSpT_Barrel;
  MonitorElement*  mNHFracVSpT_Barrel;
  MonitorElement*  mPhFracVSpT_Barrel;
  MonitorElement*  mElFracVSpT_Barrel;
  MonitorElement*  mMuFracVSpT_Barrel;
  MonitorElement*  mCHFracVSpT_EndCap;
  MonitorElement*  mNHFracVSpT_EndCap;
  MonitorElement*  mPhFracVSpT_EndCap;
  MonitorElement*  mElFracVSpT_EndCap;
  MonitorElement*  mMuFracVSpT_EndCap;
  MonitorElement*  mHFHFracVSpT_Forward;
  MonitorElement*  mHFEFracVSpT_Forward;

  MonitorElement*  mCHFracVSeta_lowPt;
  MonitorElement*  mNHFracVSeta_lowPt;
  MonitorElement*  mPhFracVSeta_lowPt;
  MonitorElement*  mElFracVSeta_lowPt;
  MonitorElement*  mMuFracVSeta_lowPt;
  MonitorElement*  mCHFracVSeta_mediumPt;
  MonitorElement*  mNHFracVSeta_mediumPt;
  MonitorElement*  mPhFracVSeta_mediumPt;
  MonitorElement*  mElFracVSeta_mediumPt;
  MonitorElement*  mMuFracVSeta_mediumPt;
  MonitorElement*  mCHFracVSeta_highPt;
  MonitorElement*  mNHFracVSeta_highPt;
  MonitorElement*  mPhFracVSeta_highPt;
  MonitorElement*  mElFracVSeta_highPt;
  MonitorElement*  mMuFracVSeta_highPt;

  MonitorElement* mCHFrac_lowPt_EndCap;
  MonitorElement* mNHFrac_lowPt_EndCap;
  MonitorElement* mPhFrac_lowPt_EndCap;
  MonitorElement* mElFrac_lowPt_EndCap;
  MonitorElement* mMuFrac_lowPt_EndCap;
  MonitorElement* mCHFrac_mediumPt_EndCap;
  MonitorElement* mNHFrac_mediumPt_EndCap;
  MonitorElement* mPhFrac_mediumPt_EndCap;
  MonitorElement* mElFrac_mediumPt_EndCap;
  MonitorElement* mMuFrac_mediumPt_EndCap;
  MonitorElement* mCHFrac_highPt_EndCap;
  MonitorElement* mNHFrac_highPt_EndCap;
  MonitorElement* mPhFrac_highPt_EndCap;
  MonitorElement* mElFrac_highPt_EndCap;
  MonitorElement* mMuFrac_highPt_EndCap;

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

  MonitorElement*   mChMultiplicity_lowPt_EndCap;
  MonitorElement*   mNeuMultiplicity_lowPt_EndCap;
  MonitorElement*   mMuMultiplicity_lowPt_EndCap;
  MonitorElement*   mChMultiplicity_mediumPt_EndCap;
  MonitorElement*   mNeuMultiplicity_mediumPt_EndCap;
  MonitorElement*   mMuMultiplicity_mediumPt_EndCap;
  MonitorElement*   mChMultiplicity_highPt_EndCap;
  MonitorElement*   mNeuMultiplicity_highPt_EndCap;
  MonitorElement*   mMuMultiplicity_highPt_EndCap;


  MonitorElement* mPt_EndCap;
  MonitorElement* mPhi_EndCap;

  MonitorElement* mPt_Forward;
  MonitorElement* mPhi_Forward;

  MonitorElement*mHFEFrac_lowPt_Forward;
  MonitorElement*mHFHFrac_lowPt_Forward;
  MonitorElement*mHFEFrac_mediumPt_Forward;
  MonitorElement*mHFHFrac_mediumPt_Forward;
  MonitorElement*mHFEFrac_highPt_Forward;
  MonitorElement*mHFHFrac_highPt_Forward;
  MonitorElement*mHFEEn_lowPt_Forward;
  MonitorElement*mHFHEn_lowPt_Forward;
  MonitorElement*mHFEEn_mediumPt_Forward;
  MonitorElement*mHFHEn_mediumPt_Forward;
  MonitorElement*mHFEEn_highPt_Forward;
  MonitorElement*mHFHEn_highPt_Forward;
  MonitorElement*   mChMultiplicity_lowPt_Forward;
  MonitorElement*   mNeuMultiplicity_lowPt_Forward;
  MonitorElement*   mMuMultiplicity_lowPt_Forward;
  MonitorElement*   mChMultiplicity_mediumPt_Forward;
  MonitorElement*   mNeuMultiplicity_mediumPt_Forward;
  MonitorElement*   mMuMultiplicity_mediumPt_Forward;
  MonitorElement*   mChMultiplicity_highPt_Forward;
  MonitorElement*   mNeuMultiplicity_highPt_Forward;
  MonitorElement*   mMuMultiplicity_highPt_Forward;

  //MonitorElement* mPt_Barrel_Lo;
  //MonitorElement* mPhi_Barrel_Lo;
  MonitorElement* mConstituents_Barrel;
  MonitorElement* mHFrac_Barrel;
  MonitorElement* mEFrac_Barrel;
  //MonitorElement* mPt_EndCap_Lo;
  //MonitorElement* mPhi_EndCap_Lo;
  MonitorElement* mConstituents_EndCap;
  MonitorElement* mHFrac_EndCap;
  MonitorElement* mEFrac_EndCap;
  //MonitorElement* mPt_Forward_Lo;
  //MonitorElement* mPhi_Forward_Lo;
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
  // ---


  //MonitorElement* mE_Barrel;
  //MonitorElement* mE_EndCap;
  //MonitorElement* mE_Forward;

  //MonitorElement* mE;
  //MonitorElement* mP;
  //MonitorElement* mMass;
  MonitorElement* mNJets;
  MonitorElement* mDPhi;

  // Leading Jet Parameters
  MonitorElement* mEtaFirst;
  MonitorElement* mPhiFirst;
  //MonitorElement* mEFirst;
  MonitorElement* mPtFirst;

  // Events passing the jet triggers
  //MonitorElement* mEta_Lo;
  MonitorElement* mPhi_Lo;
  MonitorElement* mPt_Lo;

  MonitorElement* mEta_Hi;
  MonitorElement* mPhi_Hi;
  MonitorElement* mPt_Hi;
  // PFlowJet specific

  MonitorElement* mChargedHadronEnergy;
  MonitorElement* mNeutralHadronEnergy;
  MonitorElement* mChargedEmEnergy;
  MonitorElement* mChargedMuEnergy;
  MonitorElement* mNeutralEmEnergy;
  MonitorElement* mChargedMultiplicity;
  MonitorElement* mNeutralMultiplicity;
  MonitorElement* mMuonMultiplicity;

  //new Plots with Res./ Eff. as function of neutral, charged &  em fraction

  MonitorElement* mNeutralFraction;
  MonitorElement* mNeutralFraction2;

  MonitorElement* mEEffNeutralFraction;
  MonitorElement* mEEffChargedFraction;
  MonitorElement* mEResNeutralFraction;
  MonitorElement* mEResChargedFraction;
  MonitorElement* nEEff;

  MonitorElement* mLooseJIDPassFractionVSeta;
  MonitorElement* mLooseJIDPassFractionVSpt;
  MonitorElement* mTightJIDPassFractionVSeta;
  MonitorElement* mTightJIDPassFractionVSpt;

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

  MonitorElement* mChargedHadronEnergy_profile;
  MonitorElement* mNeutralHadronEnergy_profile;
  MonitorElement* mChargedEmEnergy_profile;
  MonitorElement* mChargedMuEnergy_profile;
  MonitorElement* mNeutralEmEnergy_profile;
  MonitorElement* mChargedMultiplicity_profile;
  MonitorElement* mNeutralMultiplicity_profile;
  MonitorElement* mMuonMultiplicity_profile;
};


#endif
