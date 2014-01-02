#ifndef METAnalyzer_H
#define METAnalyzer_H


/** \class METAnalyzer
 *
 *  DQM monitoring source for MET (Mu corrected/TcMET)
 *
 *  \author A.Apresyan - Caltech
 */


#include <memory>
#include <fstream>
#include "TMath.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
//
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
//
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/JPTJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "RecoMET/METAlgorithms/interface/HcalNoiseRBXArray.h"
#include "DataFormats/METReco/interface/BeamHaloSummary.h"
#include "RecoJets/JetProducers/interface/JetIDHelper.h"

#include "DataFormats/MuonReco/interface/MuonMETCorrectionData.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonMETCorrectionData.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/PFJet.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/METCollection.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DQMOffline/JetMET/interface/JetMETDQMDCSFilter.h"
#include "CommonTools/RecoAlgos/interface/HBHENoiseFilter.h"

//class METAnalyzer : public METAnalyzerBase {
class METAnalyzer : public edm::EDAnalyzer{
 public:

  /// Constructor
  METAnalyzer(const edm::ParameterSet&/*, edm::ConsumesCollector&&*/);

  /// Destructor
  virtual ~METAnalyzer();

  /// Inizialize parameters for histo binning
  void beginJob(/*DQMStore * dbe*/);

  /// Finish up a job
  void endJob();

  // Book MonitorElements
  void bookMESet(std::string);
  void bookMonitorElement(std::string, bool);
  void bookMonitorElementTriggered(std::string, bool);

  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&);

  /// Initialize run-based parameters
  void beginRun(const edm::Run&,  const edm::EventSetup&);

  /// Finish up a run
  void endRun(const edm::Run& iRun, const edm::EventSetup& iSetup, DQMStore *dbe);

  // Fill MonitorElements
  void fillMESet(const edm::Event&, std::string, const reco::MET&, const reco::PFMET&, const reco::CaloMET&);
  void fillMonitorElement(const edm::Event&, std::string, std::string, const reco::MET&, const reco::PFMET&, const reco::CaloMET& ,bool);
  void makeRatePlot(std::string, double);

  bool selectHighPtJetEvent(const edm::Event&);
  bool selectLowPtJetEvent(const edm::Event&);
  bool selectWElectronEvent(const edm::Event&);
  bool selectWMuonEvent(const edm::Event&);

  int evtCounter;

 private:
  // ----------member data ---------------------------

  edm::ParameterSet parameters;
  // Switch for verbosity
  int _verbose;

  //edm::ConsumesCollector iC;

  DQMStore * dbe;

  std::string MetType;
  std::string mOutputFile;

  std::string _FolderName;

  edm::InputTag theMETCollectionLabel;
  edm::InputTag HcalNoiseRBXCollectionTag;
  edm::InputTag theJetCollectionLabel;
  edm::InputTag thePfJetCollectionLabel;
  edm::InputTag BeamHaloSummaryTag;
  edm::InputTag HBHENoiseFilterResultTag;
  edm::InputTag vertexTag;
  edm::InputTag gtTag;

  edm::EDGetTokenT<std::vector<reco::Vertex>>vertexToken_;
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord>gtToken_;
  edm::EDGetTokenT<reco::CaloJetCollection> caloJetsToken_;
  edm::EDGetTokenT<reco::PFJetCollection> pfJetsToken_;
  edm::EDGetTokenT<reco::JPTJetCollection> jptJetsToken_;

  edm::EDGetTokenT<bool> HBHENoiseFilterResultToken_;
  edm::EDGetTokenT<reco::BeamHaloSummary> BeamHaloSummaryToken_;

  edm::EDGetTokenT<reco::METCollection> tcMetToken_; 
  edm::EDGetTokenT<reco::PFMETCollection> pfMetToken_;
  edm::EDGetTokenT<reco::CaloMETCollection> caloMetToken_;
  edm::EDGetTokenT<reco::HcalNoiseRBXCollection> HcalNoiseRBXToken_; 




  edm::InputTag inputTrackLabel;
  edm::InputTag inputMuonLabel;
  edm::InputTag inputElectronLabel;
  edm::InputTag inputBeamSpotLabel;
  edm::InputTag inputTCMETValueMap;

  edm::EDGetTokenT<edm::View <reco::Track> >TrackToken_;
  edm::EDGetTokenT<reco::MuonCollection> MuonToken_;
  edm::EDGetTokenT<edm::View <reco::GsfElectron> >ElectronToken_;
  edm::EDGetTokenT<reco::BeamSpot> BeamspotToken_;

  edm::EDGetTokenT<edm::ValueMap<reco::MuonMETCorrectionData>> tcMET_ValueMapToken_;
  edm::Handle< edm::ValueMap<reco::MuonMETCorrectionData> > tcMet_ValueMap_Handle;

  edm::Handle< reco::MuonCollection > muon_h;
  edm::Handle< edm::View<reco::Track> > track_h;
  edm::Handle< edm::View<reco::GsfElectron > > electron_h;
  edm::Handle< reco::BeamSpot > beamSpot_h;


  edm::InputTag theTriggerResultsLabel;
  edm::EDGetTokenT<edm::TriggerResults>triggerResultsToken_;

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

  int nbinsPV;

  double PVlow; 
  double PVup;


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


  double _highPtJetThreshold;
  double _lowPtJetThreshold;
  double _highMETThreshold;
  //  double _lowMETThreshold;
  //do low MET for caloMET too?
  double _lowPFMETThreshold;

  int _numPV;
  // Et threshold for MET plots
  double _etThreshold;

  // HF calibration factor (in 31X applied by TcProducer)
  //delete altogether not used anymore
  double hfCalibFactor_;  //

  // JetID helper
  reco::helper::JetIDHelper *jetID;


  // DCS filter
  JetMETDQMDCSFilter *DCSFilter;

  //
  bool _allhist;
  bool _allSelection;
  bool _cleanupSelection;

  //
  std::vector<std::string> _FolderNames;

  //
  math::XYZPoint bspot;



  //
  DQMStore *_dbe;

  //trigger histos
  // lines commented out have been removed to improve the bin usage of JetMET DQM

  //for all MET types

  MonitorElement* hTriggerName_HighPtJet;
  MonitorElement* hTriggerName_LowPtJet;
  MonitorElement* hTriggerName_MinBias;
  MonitorElement* hTriggerName_HighMET;
  //  MonitorElement* hTriggerName_LowMET;
  MonitorElement* hTriggerName_Ele;
  MonitorElement* hTriggerName_Muon;
  MonitorElement* hMETRate;
  //only in for PF
  MonitorElement* meTriggerName_PhysDec;

  //for all MET types
  MonitorElement* hmetME;
  //MonitorElement* hNevents;
  MonitorElement* hMEx;
  MonitorElement* hMEy;
  //MonitorElement* hEz;
  MonitorElement* hMETSig;
  MonitorElement* hMET;
  MonitorElement* hMETPhi;
  MonitorElement* hSumET;

  MonitorElement* hMExLS;
  MonitorElement* hMEyLS;

  MonitorElement* hMET_logx;
  MonitorElement* hSumET_logx;

  //CaloMET specific stuff
  MonitorElement* hCaloMETPhi020;

  MonitorElement* hCaloMaxEtInEmTowers;
  MonitorElement* hCaloMaxEtInHadTowers;
  MonitorElement* hCaloEtFractionHadronic;
  MonitorElement* hCaloEmEtFraction;

  //MonitorElement* hCaloEmEtFraction002;
  //MonitorElement* hCaloEmEtFraction010;
  MonitorElement* hCaloEmEtFraction020;

  MonitorElement* hCaloHadEtInHB;
  MonitorElement* hCaloHadEtInHO;
  MonitorElement* hCaloHadEtInHE;
  MonitorElement* hCaloHadEtInHF;
  MonitorElement* hCaloEmEtInHF;
  MonitorElement* hCaloEmEtInEE;
  MonitorElement* hCaloEmEtInEB;

  MonitorElement* hCaloEmMEx;
  MonitorElement* hCaloEmMEy;
  //MonitorElement* hCaloEmEz;
  MonitorElement* hCaloEmMET;
  MonitorElement* hCaloEmMETPhi;
  //MonitorElement* hCaloEmSumET;

  MonitorElement* hCaloHaMEx;
  MonitorElement* hCaloHaMEy;
  //MonitorElement* hCaloHaEz;
  MonitorElement* hCaloHaMET;
  MonitorElement* hCaloHaMETPhi;
  //MonitorElement* hCaloHaSumET;

  MonitorElement* hCalomuPt;
  MonitorElement* hCalomuEta;
  MonitorElement* hCalomuNhits;
  MonitorElement* hCalomuChi2;
  MonitorElement* hCalomuD0;
  MonitorElement* hCaloMExCorrection;
  MonitorElement* hCaloMEyCorrection;
  MonitorElement* hCaloMuonCorrectionFlag;

  //muon corrected met, TCMet --> do we want those
  MonitorElement* hMETIonFeedbck;
  MonitorElement* hMETHPDNoise;
  MonitorElement* hMETRBXNoise;


  //is filled for TCMET
  MonitorElement* htrkPt;
  MonitorElement* htrkEta;
  MonitorElement* htrkNhits;
  MonitorElement* htrkChi2;
  MonitorElement* htrkD0;
  MonitorElement* helePt;
  MonitorElement* heleEta;
  MonitorElement* heleHoE;
  MonitorElement* hmuPt;
  MonitorElement* hmuEta;
  MonitorElement* hmuNhits;
  MonitorElement* hmuChi2;
  MonitorElement* hmuD0;

  MonitorElement* hMExCorrection;
  MonitorElement* hMEyCorrection;
  MonitorElement* hMuonCorrectionFlag;

  //now PF only things
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


  // NPV profiles --> 
  //----------------------------------------------------------------------------
  MonitorElement* meMEx_profile;
  MonitorElement* meMEy_profile;
  MonitorElement* meMET_profile;
  MonitorElement* meSumET_profile;

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

  bool isCaloMet;
  bool isTCMet;
  bool isPFMet;

};
#endif
