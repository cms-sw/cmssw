#ifndef PFJetAnalyzer_H
#define PFJetAnalyzer_H


/** \class PFJetAnalyzer
 *
 *  DQM monitoring source for PFlow Jets
 *
 *  $Date: 2010/03/04 16:31:52 $
 *  $Revision: 1.5 $
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



class PFJetAnalyzer : public PFJetAnalyzerBase {
 public:

  /// Constructor
  PFJetAnalyzer(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~PFJetAnalyzer();

  /// Inizialize parameters for histo binning
  void beginJob(DQMStore * dbe);

  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&, const reco::PFJetCollection& pfJets);
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

  MonitorElement* mPt_EndCap;
  MonitorElement* mPhi_EndCap;

  MonitorElement* mPt_Forward;
  MonitorElement* mPhi_Forward;

  MonitorElement* mPt_Barrel_Lo;
  MonitorElement* mPhi_Barrel_Lo;
  MonitorElement* mConstituents_Barrel_Lo;
  MonitorElement* mHFrac_Barrel_Lo;
  MonitorElement* mPt_EndCap_Lo;
  MonitorElement* mPhi_EndCap_Lo;
  MonitorElement* mConstituents_EndCap_Lo;
  MonitorElement* mHFrac_EndCap_Lo;
  MonitorElement* mPt_Forward_Lo;
  MonitorElement* mPhi_Forward_Lo;
  MonitorElement* mConstituents_Forward_Lo;
  MonitorElement* mHFrac_Forward_Lo;

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

  // Events passing the jet triggers
  MonitorElement* mEta_Lo;
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

};
#endif
