#ifndef METAnalyzer_H
#define METAnalyzer_H

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
#include "PhysicsTools/SelectorUtils/interface/JetIDSelectionFunctor.h"
#include "PhysicsTools/SelectorUtils/interface/PFJetIDSelectionFunctor.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"



class METAnalyzer : public edm::EDAnalyzer{
 public:

  /// Constructor
  METAnalyzer(const edm::ParameterSet&);

  /// Destructor
  virtual ~METAnalyzer();

  /// Finish up a job
  void endJob();

  // This is a temporary fix to make sure we do not have a non thread safe
  // analyzer using the thread aware DQM Analyzer base class.
  void beginRun(edm::Run const &run, edm::EventSetup const &es) override;
/// Inizialize parameters for histo binning
//  void beginJob(void);
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &);

  // Book MonitorElements
  //void bookMESet(std::string);
  //void bookMonitorElement(std::string, bool);

  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&);

  /// Initialize run-based parameters
  void dqmBeginRun(const edm::Run&,  const edm::EventSetup&);

  /// Finish up a run
  void endRun(const edm::Run& iRun, const edm::EventSetup& iSetup);
  //  void endRun(const edm::Run& iRun, const edm::EventSetup& iSetup);
  // Fill MonitorElements
  void fillMESet(const edm::Event&, std::string, const reco::MET&, const reco::PFMET&, const reco::CaloMET&);
  void fillMonitorElement(const edm::Event&, std::string, std::string, const reco::MET&, const reco::PFMET&, const reco::CaloMET& ,bool);
  void makeRatePlot(std::string, double);

//  bool selectHighPtJetEvent(const edm::Event&);
//  bool selectLowPtJetEvent(const edm::Event&);
//  bool selectWElectronEvent(const edm::Event&);
//  bool selectWMuonEvent(const edm::Event&);

 private:

 // Book MonitorElements
  void bookMESet(std::string,DQMStore::IBooker &);
// Book MonitorElements
  void bookMonitorElement(std::string,DQMStore::IBooker &, bool );

  // ----------member data ---------------------------
  edm::ParameterSet parameters;
  // Switch for verbosity
  int verbose_;

  //edm::ConsumesCollector iC;

  DQMStore * dbe_;

  std::string MetType_;
  bool outputMEsInRootFile;
  std::string mOutputFile_;
  std::string FolderName_;

  edm::InputTag metCollectionLabel_;
  edm::InputTag hcalNoiseRBXCollectionTag_;
  edm::InputTag jetCollectionLabel_;
  edm::InputTag beamHaloSummaryTag_;
  edm::InputTag hbheNoiseFilterResultTag_;
  edm::InputTag vertexTag_;
  edm::InputTag gtTag_;

  edm::EDGetTokenT<std::vector<reco::Vertex>>     vertexToken_;
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord>  gtToken_;
  edm::EDGetTokenT<reco::CaloJetCollection>       caloJetsToken_;
  edm::EDGetTokenT<reco::PFJetCollection>         pfJetsToken_;
  edm::EDGetTokenT<reco::JPTJetCollection>        jptJetsToken_;

  edm::EDGetTokenT<bool>                          hbheNoiseFilterResultToken_;
  edm::EDGetTokenT<reco::BeamHaloSummary>         beamHaloSummaryToken_;

  edm::EDGetTokenT<reco::METCollection>           tcMetToken_; 
  edm::EDGetTokenT<reco::PFMETCollection>         pfMetToken_;
  edm::EDGetTokenT<reco::CaloMETCollection>       caloMetToken_;
  edm::EDGetTokenT<reco::HcalNoiseRBXCollection>  HcalNoiseRBXToken_; 

  edm::InputTag inputTrackLabel_;
  edm::InputTag inputMuonLabel_;
  edm::InputTag inputElectronLabel_;
  edm::InputTag inputBeamSpotLabel_;
  edm::InputTag inputTCMETValueMap_;

  edm::EDGetTokenT<edm::View <reco::Track> >        TrackToken_;
  edm::EDGetTokenT<reco::MuonCollection>            MuonToken_;
  edm::EDGetTokenT<edm::View <reco::GsfElectron> >  ElectronToken_;
  edm::EDGetTokenT<reco::BeamSpot>                  BeamspotToken_;

  edm::InputTag inputJetIDValueMap;
  edm::EDGetTokenT<edm::ValueMap <reco::JetID> >jetID_ValueMapToken_;

  JetIDSelectionFunctor jetIDFunctorLoose;
  PFJetIDSelectionFunctor pfjetIDFunctorLoose;

  std::string jetCorrectionService_;

  double ptThreshold_;

 

  edm::EDGetTokenT<edm::ValueMap<reco::MuonMETCorrectionData>> tcMETValueMapToken_;
  edm::Handle< edm::ValueMap<reco::MuonMETCorrectionData> > tcMetValueMapHandle_;

  edm::Handle< reco::MuonCollection >           muonHandle_;
  edm::Handle< edm::View<reco::Track> >         trackHandle_;
  edm::Handle< edm::View<reco::GsfElectron > >  electronHandle_;
  edm::Handle< reco::BeamSpot >                 beamSpotHandle_;

  HLTConfigProvider hltConfig_;
  edm::InputTag                         triggerResultsLabel_;
  edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;

  // list of Jet or MB HLT triggers
//  std::vector<std::string > HLTPathsJetMBByName_;
  std::vector<std::string > allTriggerNames_;
  std::vector< int > allTriggerDecisions_;

  edm::VParameterSet triggerSelectedSubFolders_;
  std::vector<GenericTriggerEventFlag *>  triggerFolderEventFlag_;
  std::vector<std::vector<std::string> >  triggerFolderExpr_;
  std::vector<std::string >               triggerFolderLabels_;
  std::vector<int>                        triggerFolderDecisions_;
//  std::vector<MonitorElement* >           triggerFolderME_;

//  GenericTriggerEventFlag * highPtJetEventFlag_;
//  GenericTriggerEventFlag * lowPtJetEventFlag_;
//  GenericTriggerEventFlag * minBiasEventFlag_;
//  GenericTriggerEventFlag * highMETEventFlag_;
////GenericTriggerEventFlag * lowMETEventFlag_;
//  GenericTriggerEventFlag * eleEventFlag_;
//  GenericTriggerEventFlag * muonEventFlag_;
//
//  std::vector<std::string> highPtJetExpr_;
//  std::vector<std::string> lowPtJetExpr_;
//  std::vector<std::string> highMETExpr_;
//  //  std::vector<std::string> lowMETExpr_;
//  std::vector<std::string> muonExpr_;
//  std::vector<std::string> elecExpr_;
//  std::vector<std::string> minbiasExpr_;
//  MonitorElement* hTriggerName_HighPtJet;
//  MonitorElement* hTriggerName_LowPtJet;
//  MonitorElement* hTriggerName_MinBias;
//  MonitorElement* hTriggerName_HighMET;
//  //  MonitorElement* hTriggerName_LowMET;
//  MonitorElement* hTriggerName_Ele;
//  MonitorElement* hTriggerName_Muon;
  MonitorElement* hMETRate;

  edm::ParameterSet cleaningParameters_;
  std::string hltPhysDec_;

  int    nbinsPV_;
  double nPVMin_; 
  double nPVMax_;


  int LSBegin_;
  int LSEnd_;

  bool bypassAllPVChecks_;
  bool bypassAllDCSChecks_;
  bool runcosmics_;


//  int trigJetMB_;
//  int trigHighPtJet_;
//  int trigLowPtJet_;
//  int trigMinBias_;
//  int trigHighMET_;
////int trigLowMET_;
//  int trigEle_;
//  int trigMuon_;
//  int trigPhysDec_;

//  double highPtJetThreshold_;
//  double lowPtJetThreshold_;
//  double highMETThreshold_;

  int numPV_;
  // Et threshold for MET plots
//  double etThreshold_;

  // HF calibration factor (in 31X applied by TcProducer)
  //delete altogether not used anymore
  double hfCalibFactor_;  //

  // DCS filter
  JetMETDQMDCSFilter *DCSFilter_;

  std::vector<std::string> folderNames_;
  //
  math::XYZPoint beamSpot_;

  //trigger histos
  // lines commented out have been removed to improve the bin usage of JetMET DQM

  //for all MET types
  bool hTriggerLabelsIsSet_;
  //only in for PF
//  MonitorElement* meTriggerName_PhysDec;

  MonitorElement* lumisecME;
  MonitorElement* hTrigger;
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

  bool isCaloMet_;
  bool isTCMet_;
  bool isPFMet_;

};
#endif
