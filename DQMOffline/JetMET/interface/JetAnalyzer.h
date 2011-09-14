#ifndef JetAnalyzer_H
#define JetAnalyzer_H


/** \class JetAnalyzer
 *
 *  DQM monitoring source for Calo Jets
 *
 *  $Date: 2010/10/15 13:49:54 $
 *  $Revision: 1.13 $
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
	       const reco::CaloJetCollection& caloJets);

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

  MonitorElement* mConstituents_Barrel;
  MonitorElement* mHFrac_Barrel;
  MonitorElement* mEFrac_Barrel;
  //removed for optimization//MonitorElement* mPt_Barrel_Lo;
  //removed for optimization//MonitorElement* mPhi_Barrel_Lo;
  //removed for optimization//MonitorElement* mConstituents_Barrel_Lo;
  //removed for optimization//MonitorElement* mHFrac_Barrel_Lo;
  MonitorElement* mConstituents_EndCap;
  MonitorElement* mHFrac_EndCap;
  MonitorElement* mEFrac_EndCap;
  //removed for optimization//MonitorElement* mPt_EndCap_Lo;
  //removed for optimization//MonitorElement* mPhi_EndCap_Lo;
  //removed for optimization//MonitorElement* mConstituents_EndCap_Lo;
  //removed for optimization//MonitorElement* mHFrac_EndCap_Lo;
  MonitorElement* mConstituents_Forward;
  MonitorElement* mHFrac_Forward;
  MonitorElement* mEFrac_Forward;
  //removed for optimization//MonitorElement* mPt_Forward_Lo;
  //removed for optimization//MonitorElement* mPhi_Forward_Lo;
  //removed for optimization//MonitorElement* mConstituents_Forward_Lo;
  //removed for optimization//MonitorElement* mHFrac_Forward_Lo;

  MonitorElement* mPt_Barrel_Hi;
  MonitorElement* mPhi_Barrel_Hi;
  //removed for optimization//MonitorElement* mConstituents_Barrel_Hi;
  //removed for optimization//MonitorElement* mHFrac_Barrel_Hi;
  MonitorElement* mPt_EndCap_Hi;
  MonitorElement* mPhi_EndCap_Hi;
  //removed for optimization//MonitorElement* mConstituents_EndCap_Hi;
  //removed for optimization//MonitorElement* mHFrac_EndCap_Hi;
  MonitorElement* mPt_Forward_Hi;
  MonitorElement* mPhi_Forward_Hi;
  //removed for optimization//MonitorElement* mConstituents_Forward_Hi;
  //removed for optimization//MonitorElement* mHFrac_Forward_Hi;
  // ---


  //removed for optimizations//MonitorElement* mE_Barrel;
  //removed for optimizations//MonitorElement* mE_EndCap;
  //removed for optimizations//MonitorElement* mE_Forward;

  //removed for optimizations//MonitorElement* mE;
  //removed for optimizations//MonitorElement* mP;
  //removed for optimizations//MonitorElement* mMass;
  MonitorElement* mNJets;
  MonitorElement* mDPhi;

  // Leading Jet Parameters
  MonitorElement* mEtaFirst;
  MonitorElement* mPhiFirst;
  //removed for optimizations//MonitorElement* mEFirst;
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
  //removed for optimizations//MonitorElement* msigmaEta;
  //removed for optimizations//MonitorElement* msigmaPhi;
  MonitorElement* mLooseJIDPassFractionVSeta;
  MonitorElement* mLooseJIDPassFractionVSpt;
  MonitorElement* mTightJIDPassFractionVSeta;
  MonitorElement* mTightJIDPassFractionVSpt;

  // Events passing the jet triggers
  MonitorElement* mEta_Lo;
  MonitorElement* mPhi_Lo;
  MonitorElement* mPt_Lo;

  MonitorElement* mEta_Hi;
  MonitorElement* mPhi_Hi;
  MonitorElement* mPt_Hi;

};
#endif
