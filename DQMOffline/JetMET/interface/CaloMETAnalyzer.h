#ifndef CaloMETAnalyzer_H
#define CaloMETAnalyzer_H

/** \class CaloMETAnalyzer
 *
 *  DQM monitoring source for CaloMET
 *
 *  $Date: 2009/06/30 13:48:19 $
 *  $Revision: 1.1 $
 *  \author F. Chlebana - Fermilab
 *          K. Hatakeyama - Rockefeller University
 */

#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMOffline/JetMET/interface/CaloMETAnalyzerBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
//
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "RecoMET/METAlgorithms/interface/HcalNoiseRBXArray.h"
#include "DataFormats/METReco/interface/HcalNoiseSummary.h"

#include "RecoJets/JetAlgorithms/interface/JetIDHelper.h"

class CaloMETAnalyzer : public CaloMETAnalyzerBase {
 public:

  /// Constructor
  CaloMETAnalyzer(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~CaloMETAnalyzer();

  /// Inizialize parameters for histo binning
  void beginJob(edm::EventSetup const& iSetup, DQMStore *dbe);

  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&, 
               const edm::TriggerResults&);

  /// Initialize run-based parameters
  void beginRun(const edm::Run&,  const edm::EventSetup&);

  /// Finish up a run
  void endRun(const edm::Run& iRun, const edm::EventSetup& iSetup, DQMStore *dbe);

  // Book MonitorElements
  void bookMESet(std::string);
  void bookMonitorElement(std::string, bool);

  // Fill MonitorElements
  void fillMESet(const edm::Event&, std::string, const reco::CaloMET&);
  void fillMonitorElement(const edm::Event&, std::string, const reco::CaloMET&, bool);
  void makeRatePlot(std::string, double);

  void validateMET(const reco::CaloMET&, edm::Handle<edm::View<Candidate> >);

  int evtCounter;

 private:
  // ----------member data ---------------------------
  
  edm::ParameterSet parameters;
  // Switch for verbosity
  int _verbose;

  std::string metname;
  std::string _source; // HLT? FU?
  
  edm::InputTag theCaloMETCollectionLabel;

  edm::InputTag theCaloTowersLabel;
  edm::InputTag theJetCollectionLabel;
  edm::InputTag HcalNoiseRBXCollectionTag;
  edm::InputTag HcalNoiseSummaryTag;

  /// number of Jet or MB HLT trigger paths 
  unsigned int nHLTPathsJetMB_;
  // list of Jet or MB HLT triggers
  std::vector<std::string > HLTPathsJetMBByName_;
  // list of Jet or MB HLT trigger index
  std::vector<unsigned int> HLTPathsJetMBByIndex_;

  std::string _hlt_HighPtJet;
  std::string _hlt_LowPtJet;
  std::string _hlt_HighMET;
  std::string _hlt_LowMET;
  
  int _trig_JetMB;
  int _trig_HighPtJet;
  int _trig_LowPtJet;
  int _trig_HighMET;
  int _trig_LowMET;

  HLTConfigProvider hltConfig_;
  std::string processname_;
  
  // Et threshold for MET plots
  double _etThreshold;

  // JetID helper
  reco::helper::JetID jetID;

  // HCalNoise

  //
  bool _allhist;
  bool _allSelection;

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

  //
  std::vector<std::string> _FolderNames;

  //
  DQMStore *_dbe;

  //the histos
  MonitorElement* jetME;

  MonitorElement* meNevents;
  MonitorElement* meCaloMEx;
  MonitorElement* meCaloMEy;
  MonitorElement* meCaloEz;
  MonitorElement* meCaloMETSig;
  MonitorElement* meCaloMET;
  MonitorElement* meCaloMETPhi;
  MonitorElement* meCaloSumET;
  MonitorElement* meCaloMExLS;
  MonitorElement* meCaloMEyLS;

  MonitorElement* meCaloMaxEtInEmTowers;
  MonitorElement* meCaloMaxEtInHadTowers;
  MonitorElement* meCaloEtFractionHadronic;
  MonitorElement* meCaloEmEtFraction;

  MonitorElement* meCaloHadEtInHB;
  MonitorElement* meCaloHadEtInHO;
  MonitorElement* meCaloHadEtInHE;
  MonitorElement* meCaloHadEtInHF;
  MonitorElement* meCaloHadEtInEB;
  MonitorElement* meCaloHadEtInEE;
  MonitorElement* meCaloEmEtInHF;
  MonitorElement* meCaloEmEtInEE;
  MonitorElement* meCaloEmEtInEB;

  MonitorElement* meCaloMETRate;

  MonitorElement* meCaloMEx_cut1;
  MonitorElement* meCaloMEy_cut1;
  MonitorElement* meCaloEz_cut1;
  MonitorElement* meCaloMETSig_cut1;
  MonitorElement* meCaloMET_cut1;
  MonitorElement* meCaloMETPhi_cut1;
  MonitorElement* meCaloSumET_cut1;

  MonitorElement* meCaloMETRate_cut1;

};
#endif
