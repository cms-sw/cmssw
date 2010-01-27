#ifndef CaloMETAnalyzer_H
#define CaloMETAnalyzer_H

/** \class CaloMETAnalyzer
 *
 *  DQM monitoring source for CaloMET
 *
 *  $Date: 2010/01/25 08:21:25 $
 *  $Revision: 1.12 $
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
//#include "DataFormats/METReco/interface/CaloMETCollection.h"
//#include "DataFormats/METReco/interface/CaloMET.h"
//#include "DataFormats/METReco/interface/CaloMETFwd.h"
//
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "RecoMET/METAlgorithms/interface/HcalNoiseRBXArray.h"
#include "DataFormats/METReco/interface/HcalNoiseSummary.h"
#include "DataFormats/METReco/interface/BeamHaloSummary.h"
#include "RecoJets/JetAlgorithms/interface/JetIDHelper.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

class CaloMETAnalyzer : public CaloMETAnalyzerBase {
 public:

  /// Constructor
  CaloMETAnalyzer(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~CaloMETAnalyzer();

  /// Inizialize parameters for histo binning
  void beginJob(DQMStore * dbe);

  /// Finish up a job
  void endJob();

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
  void fillMonitorElement(const edm::Event&, std::string, std::string, const reco::CaloMET&, bool);
  void makeRatePlot(std::string, double);

  void validateMET(const reco::CaloMET&, edm::Handle<edm::View<Candidate> >);
  void computeEmHaMET(edm::Handle<edm::View<Candidate> >);

  bool selectHighPtJetEvent(const edm::Event&);
  bool selectLowPtJetEvent(const edm::Event&);
  bool selectWElectronEvent(const edm::Event&);
  bool selectWMuonEvent(const edm::Event&);
  bool selectPhysicsDeclaredEvent(const edm::Event&);

  int evtCounter;

 private:
  // ----------member data ---------------------------
  
  edm::ParameterSet parameters;
  // Switch for verbosity
  int _verbose;
  int _print;

  std::string metname;
  std::string _source; // HLT? FU?
  
  edm::InputTag theCaloMETCollectionLabel;

  edm::InputTag theCaloTowersLabel;
  edm::InputTag theJetCollectionLabel;
  edm::InputTag HcalNoiseRBXCollectionTag;
  edm::InputTag HcalNoiseSummaryTag;
  edm::InputTag BeamHaloSummaryTag;

  // list of Jet or MB HLT triggers
  std::vector<std::string > HLTPathsJetMBByName_;

  std::string _hlt_HighPtJet;
  std::string _hlt_LowPtJet;
  std::string _hlt_HighMET;
  std::string _hlt_LowMET;
  std::string _hlt_Ele;
  std::string _hlt_Muon;
  std::string _hlt_PhysDec;

  std::vector<unsigned > _techTrigsAND;
  std::vector<unsigned > _techTrigsOR;
  std::vector<unsigned > _techTrigsNOT;

  bool _doPVCheck;
  bool _doHLTPhysicsOn;

  bool     _tightBHFiltering;
  int      _tightJetIDFiltering;
  bool     _tightHcalFiltering;

  int _nvtx_min;
  int _nvtxtrks_min;
  double _vtxchi2_max;
  double _vtxz_max;

  int _trig_JetMB;
  int _trig_HighPtJet;
  int _trig_LowPtJet;
  int _trig_HighMET;
  int _trig_LowMET;
  int _trig_Ele;
  int _trig_Muon;
  int _trig_PhysDec;

  double _highPtJetThreshold;
  double _lowPtJetThreshold;
  double _highMETThreshold;
  double _lowMETThreshold;

  // Et threshold for MET plots
  double _etThreshold;

  // JetID helper
  reco::helper::JetIDHelper *jetID;

  //
  bool _allhist;
  bool _allSelection;

  //
  std::vector<std::string> _FolderNames;

  //
  double _EmMEx;
  double _EmMEy;
  double _EmMET;
  double _EmCaloEz;
  double _EmSumEt;
  double _EmMetPhi;
  //
  double _HaMEx;
  double _HaMEy;
  double _HaMET;
  double _HaCaloEz;
  double _HaSumEt;
  double _HaMetPhi;

  //
  DQMStore *_dbe;

  //the histos
  MonitorElement* metME;

  MonitorElement* meTriggerName_HighPtJet;
  MonitorElement* meTriggerName_LowPtJet;
  MonitorElement* meTriggerName_HighMET;
  MonitorElement* meTriggerName_LowMET;
  MonitorElement* meTriggerName_Ele;
  MonitorElement* meTriggerName_Muon;
  MonitorElement* meTriggerName_PhysDec;

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

  MonitorElement* meCaloMET_logx;
  MonitorElement* meCaloSumET_logx;

  MonitorElement* meCaloMETIonFeedbck;
  MonitorElement* meCaloMETHPDNoise;
  MonitorElement* meCaloMETRBXNoise;

  MonitorElement* meCaloMETPhi002;
  MonitorElement* meCaloMETPhi010;
  MonitorElement* meCaloMETPhi020;

  MonitorElement* meCaloMaxEtInEmTowers;
  MonitorElement* meCaloMaxEtInHadTowers;
  MonitorElement* meCaloEtFractionHadronic;
  MonitorElement* meCaloEmEtFraction;

  MonitorElement* meCaloEmEtFraction002;
  MonitorElement* meCaloEmEtFraction010;
  MonitorElement* meCaloEmEtFraction020;

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

  //
  std::map<std::string, MonitorElement*> me;

};
#endif
