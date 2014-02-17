#ifndef JetAnalyzer_H
#define JetAnalyzer_H


/** \class JetAnalyzer
 *
 *  DQM monitoring source for Calo Jets
 *
 *  $Date: 2012/03/23 18:24:43 $
 *  $Revision: 1.17 $
 *  \author F. Chlebana - Fermilab
 */


#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMOffline/JetMET/interface/JetAnalyzerBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
//
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "RecoJets/JetProducers/interface/JetIDHelper.h"

#include <string>


#include "GlobalVariables.h"


class JetAnalyzer : public JetAnalyzerBase {
 public:

  /// Constructor
  //  JetAnalyzer(const edm::ParameterSet&, JetServiceProxy *theService);
  JetAnalyzer(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~JetAnalyzer();

  /// Inizialize parameters for histo binning
  void beginJob(DQMStore * dbe);

  /// Finish up a job
  void endJob();

  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&, 
	       const reco::CaloJetCollection& caloJets,
	       const int numPV);

  void setSource(std::string source) {
    _source = source;
  }

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
  
  edm::ParameterSet parameters;
  // Switch for verbosity
  std::string jetname;
  std::string _source;
  // Calo Jet Label
  edm::InputTag theCaloJetCollectionLabel;

  int   _JetLoPass;
  int   _JetHiPass;
  int   _leadJetFlag;
  int _theend;
  double _ptThreshold;

  double _asymmetryThirdJetCut;
  double _balanceThirdJetCut;

  int makedijetselection;

  //JID cuts
  double _fHPDMax;
  double _resEMFMin;
  int _n90HitsMin;
  //
  int fillJIDPassFrac;
  double _fHPDMaxLoose;
  double _resEMFMinLoose;
  int _n90HitsMinLoose;
  double _fHPDMaxTight;
  double _resEMFMinTight;
  int _n90HitsMinTight;
  double _sigmaEtaMinTight;
  double _sigmaPhiMinTight; 

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

  //the histos
  MonitorElement* jetME;

  // JetID helper
  reco::helper::JetIDHelper *jetID;

  // Calo Jets

  //  std::vector<MonitorElement*> etaCaloJet;
  //  std::vector<MonitorElement*> phiCaloJet;
  //  std::vector<MonitorElement*> ptCaloJet;
  //  std::vector<MonitorElement*> qGlbTrack;

  //  MonitorElement* etaCaloJet;
  //  MonitorElement* phiCaloJet;
  //  MonitorElement* ptCaloJet;

  // Generic Jet Parameters

  // --- Used for Data Certification
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

  MonitorElement* mPt_EndCap;
  MonitorElement* mPhi_EndCap;

  MonitorElement* mPt_Forward;
  MonitorElement* mPhi_Forward;

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
  //  MonitorElement* mMass;
  MonitorElement* mNJets;
  MonitorElement* mDPhi;

  // Leading Jet Parameters
  MonitorElement* mEtaFirst;
  MonitorElement* mPhiFirst;
  //MonitorElement* mEFirst;
  MonitorElement* mPtFirst;


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
  //  MonitorElement* mEnergyFractionHadronic;
  //  MonitorElement* mEnergyFractionEm;
  MonitorElement* mN90Hits;
  MonitorElement* mfHPD;
  MonitorElement* mfRBX;
  MonitorElement* mresEMF;
  //  MonitorElement* msigmaEta;
  //  MonitorElement* msigmaPhi;
  MonitorElement* mLooseJIDPassFractionVSeta;
  MonitorElement* mLooseJIDPassFractionVSpt;
  MonitorElement* mTightJIDPassFractionVSeta;
  MonitorElement* mTightJIDPassFractionVSpt;

  // Events passing the jet triggers
  //MonitorElement* mEta_Lo;
  MonitorElement* mPhi_Lo;
  MonitorElement* mPt_Lo;

  MonitorElement* mEta_Hi;
  MonitorElement* mPhi_Hi;
  MonitorElement* mPt_Hi;

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
};


#endif
