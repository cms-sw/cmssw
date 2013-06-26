#ifndef PFMETAnalyzer_H
#define PFMETAnalyzer_H


/** \class PFMETAnalyzer
 *
 *  DQM monitoring source for PFMET
 *
 *  $Date: 2012/05/20 17:57:01 $
 *  $Revision: 1.30 $
 *  \author K. Hatakeyama - Rockefeller University
 *          A.Apresyan - Caltech 
 */


#include <memory>
#include <fstream>
#include "TMath.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMOffline/JetMET/interface/PFMETAnalyzerBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include <DataFormats/ParticleFlowCandidate/interface/PFCandidate.h>

#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "RecoMET/METAlgorithms/interface/HcalNoiseRBXArray.h"
#include "DataFormats/METReco/interface/BeamHaloSummary.h"
#include "RecoJets/JetProducers/interface/JetIDHelper.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DQMOffline/JetMET/interface/JetMETDQMDCSFilter.h"


#include "GlobalVariables.h"


class PFMETAnalyzer : public PFMETAnalyzerBase {
 public:

  /// Constructor
  PFMETAnalyzer(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~PFMETAnalyzer();

  /// Inizialize parameters for histo binning
  void beginJob(DQMStore * dbe);

  /// Finish up a job
  void endJob();

  // Book MonitorElements
  void bookMESet(std::string);
  void bookMonitorElement(std::string, bool);

  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&, 
               const edm::TriggerResults&);

  /// Initialize run-based parameters
  void beginRun(const edm::Run&,  const edm::EventSetup&);

  /// Finish up a run
  void endRun(const edm::Run& iRun, const edm::EventSetup& iSetup, DQMStore *dbe);

  // Fill MonitorElements
  void fillMESet(const edm::Event&, std::string, const reco::PFMET&);
  void fillMonitorElement(const edm::Event&, std::string, std::string, const reco::PFMET&, bool);
  void makeRatePlot(std::string, double);

  void validateMET(const reco::PFMET&, edm::Handle<edm::View<reco::PFCandidate> >);

  bool selectHighPtJetEvent(const edm::Event&);
  bool selectLowPtJetEvent(const edm::Event&);
  bool selectWElectronEvent(const edm::Event&);
  bool selectWMuonEvent(const edm::Event&);

  void setSource(std::string source) {
    _source = source;
  }

  int evtCounter;

 private:
  // ----------member data ---------------------------
  
  edm::ParameterSet parameters;
  int _verbose;

  std::string metname;
  std::string _source;

  edm::InputTag thePfMETCollectionLabel;
  edm::InputTag HcalNoiseRBXCollectionTag;
  edm::InputTag theJetCollectionLabel;
  edm::InputTag thePfJetCollectionLabel;
  edm::InputTag PFCandidatesTag;
  edm::InputTag BeamHaloSummaryTag;
  edm::InputTag HBHENoiseFilterResultTag;
  edm::InputTag vertexTag;
  edm::InputTag gtTag;

  // list of Jet or MB HLT triggers
  std::vector<std::string > HLTPathsJetMBByName_;

  GenericTriggerEventFlag * _HighPtJetEventFlag;
  GenericTriggerEventFlag * _LowPtJetEventFlag;
  GenericTriggerEventFlag * _MinBiasEventFlag;
  GenericTriggerEventFlag * _HighMETEventFlag;
  //  GenericTriggerEventFlag * _LowMETEventFlag;
  GenericTriggerEventFlag * _EleEventFlag;
  GenericTriggerEventFlag * _MuonEventFlag;

  std::vector<std::string> highPtJetExpr_;
  std::vector<std::string> lowPtJetExpr_;
  std::vector<std::string> highMETExpr_;
  //  std::vector<std::string> lowMETExpr_;
  std::vector<std::string> muonExpr_;
  std::vector<std::string> elecExpr_;
  std::vector<std::string> minbiasExpr_;

  edm::ParameterSet theCleaningParameters;
  std::string _hlt_PhysDec;

  std::vector<unsigned > _techTrigsAND;
  std::vector<unsigned > _techTrigsOR;
  std::vector<unsigned > _techTrigsNOT;

  bool _doPVCheck;
  bool _doHLTPhysicsOn;

  bool     _tightBHFiltering;
  int      _tightJetIDFiltering;

  int _nvtx_min;
  int _nvtxtrks_min;
  int _vtxndof_min;
  double _vtxchi2_max;
  double _vtxz_max;
  
  int _trig_JetMB;
  int _trig_HighPtJet;
  int _trig_LowPtJet;
  int _trig_MinBias;
  int _trig_HighMET;
  //  int _trig_LowMET;
  int _trig_Ele;
  int _trig_Muon;
  int _trig_PhysDec;


  double _highPtPFJetThreshold;
  double _lowPtPFJetThreshold;
  double _highPFMETThreshold;
  double _lowPFMETThreshold;

  int _numPV;
  // Et threshold for MET plots
  double _etThreshold;

  // HF calibration factor (in 31X applied by PFProducer)
  double hfCalibFactor_;  //

  // JetID helper
  reco::helper::JetIDHelper *jetID;

  // DCS filter
  JetMETDQMDCSFilter *DCSFilter;

  //
  bool _allhist;
  bool _allSelection;
  bool _cleanupSelection;

  std::vector<std::string> _FolderNames;

  DQMStore *_dbe;


  // MonitorElements
  //----------------------------------------------------------------------------
  MonitorElement* metME;
  MonitorElement* mePfMETRate;

  MonitorElement* meTriggerName_HighPtJet;
  MonitorElement* meTriggerName_LowPtJet;
  MonitorElement* meTriggerName_MinBias;
  MonitorElement* meTriggerName_HighMET;
  MonitorElement* meTriggerName_Ele;
  MonitorElement* meTriggerName_Muon;
  MonitorElement* meTriggerName_PhysDec;

  MonitorElement* mePfMEx;
  MonitorElement* mePfMEy;
  MonitorElement* mePfMETSig;
  MonitorElement* mePfMET;
  MonitorElement* mePfMETPhi;
  MonitorElement* mePfSumET;
  MonitorElement* mePfMExLS;
  MonitorElement* mePfMEyLS;
  MonitorElement* mePfMET_logx;
  MonitorElement* mePfSumET_logx;

  MonitorElement* mePhotonEtFraction;
  MonitorElement* mePhotonEt;
  MonitorElement* meNeutralHadronEtFraction;
  MonitorElement* meNeutralHadronEt;
  MonitorElement* meElectronEtFraction;
  MonitorElement* meElectronEt;
  MonitorElement* meChargedHadronEtFraction;
  MonitorElement* meChargedHadronEt;
  MonitorElement* meMuonEtFraction;
  MonitorElement* meMuonEt;
  MonitorElement* meHFHadronEtFraction;
  MonitorElement* meHFHadronEt;
  MonitorElement* meHFEMEtFraction;
  MonitorElement* meHFEMEt;


  // NPV profiles
  //----------------------------------------------------------------------------
  MonitorElement* mePfMEx_profile;
  MonitorElement* mePfMEy_profile;
  MonitorElement* mePfMET_profile;
  MonitorElement* mePfSumET_profile;

  MonitorElement* mePhotonEtFraction_profile;
  MonitorElement* mePhotonEt_profile;
  MonitorElement* meNeutralHadronEtFraction_profile;
  MonitorElement* meNeutralHadronEt_profile;
  MonitorElement* meElectronEtFraction_profile;
  MonitorElement* meElectronEt_profile;
  MonitorElement* meChargedHadronEtFraction_profile;
  MonitorElement* meChargedHadronEt_profile;
  MonitorElement* meMuonEtFraction_profile;
  MonitorElement* meMuonEt_profile;
  MonitorElement* meHFHadronEtFraction_profile;
  MonitorElement* meHFHadronEt_profile;
  MonitorElement* meHFEMEtFraction_profile;
  MonitorElement* meHFEMEt_profile;
};


#endif
