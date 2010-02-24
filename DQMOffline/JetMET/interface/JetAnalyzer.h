#ifndef JetAnalyzer_H
#define JetAnalyzer_H


/** \class JetAnalyzer
 *
 *  DQM monitoring source for Calo Jets
 *
 *  $Date: 2010/02/24 09:24:38 $
 *  $Revision: 1.6 $
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

#include "RecoJets/JetAlgorithms/interface/JetIDHelper.h"

#include <string>
using namespace std;
using namespace edm;

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
  //jet counters
  int getNjets_HB() {
    return  _Njets_HB;
  }
  int getNjets_BE() {
    return  _Njets_BE;
  }
  int getNjets_HE() {
    return  _Njets_HE;
  }
  int getNjets_EF() {
    return  _Njets_EF;
  }
  int getNjets_HF() {
    return  _Njets_HF;
  }
  //
  void setNjets_HB(int hb) {
    _Njets_HB = hb;
  }
  void setNjets_BE(int be) {
    _Njets_BE = be;
  }
  void setNjets_HE(int he) {
    _Njets_HE = he;
  }
  void setNjets_EF(int ef) {
    _Njets_EF = ef;
  }
  void setNjets_HF(int hf) {
    _Njets_HF = hf;
  }
  //
  int getNCleanedjets_HB() {
    return  _NCleanedjets_HB;
  }
  int getNCleanedjets_BE() {
    return  _NCleanedjets_BE;
  }
  int getNCleanedjets_HE() {
    return  _NCleanedjets_HE;
  }
  int getNCleanedjets_EF() {
    return  _NCleanedjets_EF;
  }
  int getNCleanedjets_HF() {
    return  _NCleanedjets_HF;
  }
  //
  void setNCleanedjets_HB(int hb) {
    _NCleanedjets_HB = hb;
  }
  void setNCleanedjets_BE(int be) {
    _NCleanedjets_BE = be;
  }
  void setNCleanedjets_HE(int he) {
    _NCleanedjets_HE = he;
  }
  void setNCleanedjets_EF(int ef) {
    _NCleanedjets_EF = ef;
  }
  void setNCleanedjets_HF(int hf) {
    _NCleanedjets_HF = hf;
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
  double _fHPDMax;
  double _resEMFMin;
  int _n90HitsMin;

  int iscleaned;
  int makedijetselection;
  int _Njets_HB;
  int _Njets_BE;
  int _Njets_HE;
  int _Njets_EF;
  int _Njets_HF;
  int _NCleanedjets_HB;
  int _NCleanedjets_BE;
  int _NCleanedjets_HE;
  int _NCleanedjets_EF;
  int _NCleanedjets_HF;

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
  MonitorElement* mEta_Barrel;
  MonitorElement* mPhi_Barrel;

  MonitorElement* mPt_EndCap;
  MonitorElement* mEta_EndCap;
  MonitorElement* mPhi_EndCap;

  MonitorElement* mPt_Forward;
  MonitorElement* mEta_Forward;
  MonitorElement* mPhi_Forward;

  MonitorElement* mPt_Barrel_Lo;
  MonitorElement* mEta_Barrel_Lo;
  MonitorElement* mPhi_Barrel_Lo;
  MonitorElement* mConstituents_Barrel_Lo;
  MonitorElement* mHFrac_Barrel_Lo;
  MonitorElement* mPt_EndCap_Lo;
  MonitorElement* mEta_EndCap_Lo;
  MonitorElement* mPhi_EndCap_Lo;
  MonitorElement* mConstituents_EndCap_Lo;
  MonitorElement* mHFrac_EndCap_Lo;
  MonitorElement* mPt_Forward_Lo;
  MonitorElement* mEta_Forward_Lo;
  MonitorElement* mPhi_Forward_Lo;
  MonitorElement* mConstituents_Forward_Lo;
  MonitorElement* mHFrac_Forward_Lo;

  MonitorElement* mPt_Barrel_Hi;
  MonitorElement* mEta_Barrel_Hi;
  MonitorElement* mPhi_Barrel_Hi;
  MonitorElement* mConstituents_Barrel_Hi;
  MonitorElement* mHFrac_Barrel_Hi;
  MonitorElement* mPt_EndCap_Hi;
  MonitorElement* mEta_EndCap_Hi;
  MonitorElement* mPhi_EndCap_Hi;
  MonitorElement* mConstituents_EndCap_Hi;
  MonitorElement* mHFrac_EndCap_Hi;
  MonitorElement* mPt_Forward_Hi;
  MonitorElement* mEta_Forward_Hi;
  MonitorElement* mPhi_Forward_Hi;
  MonitorElement* mConstituents_Forward_Hi;
  MonitorElement* mHFrac_Forward_Hi;
  // ---


  MonitorElement* mE_Barrel;
  MonitorElement* mE_EndCap;
  MonitorElement* mE_Forward;

  MonitorElement* mE;
  MonitorElement* mP;
  MonitorElement* mMass;
  MonitorElement* mNJets;
  MonitorElement* mDPhi;

  // Leading Jet Parameters
  MonitorElement* mEtaFirst;
  MonitorElement* mPhiFirst;
  MonitorElement* mEFirst;
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
  MonitorElement* mJIDEff;

  // Events passing the jet triggers
  MonitorElement* mEta_Lo;
  MonitorElement* mPhi_Lo;
  MonitorElement* mPt_Lo;

  MonitorElement* mEta_Hi;
  MonitorElement* mPhi_Hi;
  MonitorElement* mPt_Hi;

};
#endif
