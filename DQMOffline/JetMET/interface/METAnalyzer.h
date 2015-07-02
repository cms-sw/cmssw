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
 *          M. Artur W2eber
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
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DQMOffline/JetMET/interface/JetMETDQMDCSFilter.h"
#include "PhysicsTools/SelectorUtils/interface/JetIDSelectionFunctor.h"
#include "PhysicsTools/SelectorUtils/interface/PFJetIDSelectionFunctor.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DataFormats/MuonReco/interface/Muon.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"

#include <map>
#include <string>



class METAnalyzer : public DQMEDAnalyzer{
 public:

  /// Constructor
  METAnalyzer(const edm::ParameterSet&);

  /// Destructor
  virtual ~METAnalyzer();

/// Inizialize parameters for histo binning
//  void beginJob(void);
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

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
  void fillMESet(const edm::Event&, std::string, const reco::MET&, const pat::MET&, const reco::PFMET&, const reco::CaloMET&, const reco::Candidate::PolarLorentzVector&, std::map<std::string,MonitorElement*>&,std::vector<bool>);
  void fillMonitorElement(const edm::Event&, std::string, std::string, const reco::MET&, const pat::MET&, const reco::PFMET&, const reco::CaloMET& , const reco::Candidate::PolarLorentzVector& ,std::map<std::string,MonitorElement*>&,bool,bool,std::vector<bool>);
  void makeRatePlot(std::string, double);

//  bool selectHighPtJetEvent(const edm::Event&);
//  bool selectLowPtJetEvent(const edm::Event&);
//  bool selectWElectronEvent(const edm::Event&);
//  bool selectWMuonEvent(const edm::Event&);

 private:

 // Book MonitorElements
  void bookMESet(std::string,DQMStore::IBooker &,std::map<std::string,MonitorElement*>&);
// Book MonitorElements
  void bookMonitorElement(std::string,DQMStore::IBooker &, std::map<std::string,MonitorElement*>&,bool ,bool,bool);

  // ----------member data ---------------------------
  edm::ParameterSet parameters;
  // Switch for verbosity
  int verbose_;

  std::string MetType_;
  std::string FolderName_;

  edm::InputTag metCollectionLabel_;
  edm::InputTag hcalNoiseRBXCollectionTag_;
  edm::InputTag jetCollectionLabel_;
  edm::InputTag hbheNoiseFilterResultTag_;
  edm::InputTag vertexTag_;
  edm::InputTag gtTag_;

  edm::EDGetTokenT<std::vector<reco::Vertex>>     vertexToken_;
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord>  gtToken_;
  edm::EDGetTokenT<reco::CaloJetCollection>       caloJetsToken_;
  edm::EDGetTokenT<reco::PFJetCollection>         pfJetsToken_;
  edm::EDGetTokenT<pat::JetCollection>        patJetsToken_;
  edm::EDGetTokenT<reco::MuonCollection>         MuonsToken_;

  edm::EDGetTokenT<bool>                          hbheNoiseFilterResultToken_;

  edm::EDGetTokenT<pat::METCollection>           patMetToken_; 
  edm::EDGetTokenT<reco::PFMETCollection>         pfMetToken_;
  edm::EDGetTokenT<reco::CaloMETCollection>       caloMetToken_;

  edm::InputTag inputJetIDValueMap;
  edm::EDGetTokenT<edm::ValueMap <reco::JetID> >jetID_ValueMapToken_;

  JetIDSelectionFunctor jetIDFunctorLoose;
  PFJetIDSelectionFunctor pfjetIDFunctorLoose;

 
  std::string  m_l1algoname_;
  int m_bitAlgTechTrig_;

  double ptThreshold_;

  HLTConfigProvider hltConfig_;
  edm::InputTag                         triggerResultsLabel_;
  edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;

  // list of Jet or MB HLT triggers
//  std::vector<std::string > HLTPathsJetMBByName_;
  std::vector<std::string > allTriggerNames_;
  std::vector< int > allTriggerDecisions_;

  edm::EDGetTokenT<reco::JetCorrector> jetCorrectorToken_;

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
  std::vector<edm::ParameterSet> diagnosticsParameters_;

  std::string hltPhysDec_;

  int    nbinsPV_;
  double nPVMin_; 
  double nPVMax_;


  int LSBegin_;
  int LSEnd_;

  bool bypassAllPVChecks_;
  bool bypassAllDCSChecks_;
  bool runcosmics_;
  bool onlyCleaned_;


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

 
  //now PF only things
  MonitorElement* mePhotonEtFraction;
  MonitorElement* mePhotonEt;
  MonitorElement* meNeutralHadronEtFraction;
  MonitorElement* meNeutralHadronEt;
  MonitorElement* meElectronEt;
  MonitorElement* meChargedHadronEtFraction;
  MonitorElement* meChargedHadronEt;
  MonitorElement* meMuonEt;
  MonitorElement* meHFHadronEtFraction;
  MonitorElement* meHFHadronEt;
  MonitorElement* meHFEMEtFraction;
  MonitorElement* meHFEMEt;
 //MEs where we fill if the previous two bunches are empty (25 ns bunch spacing)
  MonitorElement* mePhotonEtFraction_BXm2BXm1Empty;
  MonitorElement* meNeutralHadronEtFraction_BXm2BXm1Empty;
  MonitorElement* meChargedHadronEtFraction_BXm2BXm1Empty;
  MonitorElement* meMET_BXm2BXm1Empty;
  MonitorElement* meSumET_BXm2BXm1Empty;

  MonitorElement* meMETPhiChargedHadronsBarrel_BXm2BXm1Empty;
  MonitorElement* meMETPhiChargedHadronsEndcapPlus_BXm2BXm1Empty;
  MonitorElement* meMETPhiChargedHadronsEndcapMinus_BXm2BXm1Empty;
  MonitorElement* meMETPhiNeutralHadronsBarrel_BXm2BXm1Empty;
  MonitorElement* meMETPhiNeutralHadronsEndcapPlus_BXm2BXm1Empty;
  MonitorElement* meMETPhiNeutralHadronsEndcapMinus_BXm2BXm1Empty;
  MonitorElement* meMETPhiPhotonsBarrel_BXm2BXm1Empty;
  MonitorElement* meMETPhiPhotonsEndcapPlus_BXm2BXm1Empty;
  MonitorElement* meMETPhiPhotonsEndcapMinus_BXm2BXm1Empty;
  MonitorElement* meMETPhiHFHadronsPlus_BXm2BXm1Empty;
  MonitorElement* meMETPhiHFHadronsMinus_BXm2BXm1Empty;
  MonitorElement* meMETPhiHFEGammasPlus_BXm2BXm1Empty;
  MonitorElement* meMETPhiHFEGammasMinus_BXm2BXm1Empty;

  //MEs where we fill if the previous bunch is empty (25 ns bunch spacing)
  MonitorElement* mePhotonEtFraction_BXm1Empty;
  MonitorElement* meNeutralHadronEtFraction_BXm1Empty;
  MonitorElement* meChargedHadronEtFraction_BXm1Empty;
  MonitorElement* meMET_BXm1Empty;
  MonitorElement* meSumET_BXm1Empty;

  MonitorElement* meMETPhiChargedHadronsBarrel_BXm1Empty;
  MonitorElement* meMETPhiChargedHadronsEndcapPlus_BXm1Empty;
  MonitorElement* meMETPhiChargedHadronsEndcapMinus_BXm1Empty;
  MonitorElement* meMETPhiNeutralHadronsBarrel_BXm1Empty;
  MonitorElement* meMETPhiNeutralHadronsEndcapPlus_BXm1Empty;
  MonitorElement* meMETPhiNeutralHadronsEndcapMinus_BXm1Empty;
  MonitorElement* meMETPhiPhotonsBarrel_BXm1Empty;
  MonitorElement* meMETPhiPhotonsEndcapPlus_BXm1Empty;
  MonitorElement* meMETPhiPhotonsEndcapMinus_BXm1Empty;
  MonitorElement* meMETPhiHFHadronsPlus_BXm1Empty;
  MonitorElement* meMETPhiHFHadronsMinus_BXm1Empty;
  MonitorElement* meMETPhiHFEGammasPlus_BXm1Empty;
  MonitorElement* meMETPhiHFEGammasMinus_BXm1Empty;

  //MEs where we fill if the previous bunch is filled (25 ns bunch spacing)
  MonitorElement* mePhotonEtFraction_BXm1Filled;
  MonitorElement* meNeutralHadronEtFraction_BXm1Filled;
  MonitorElement* meChargedHadronEtFraction_BXm1Filled;
  MonitorElement* meMET_BXm1Filled;
  MonitorElement* meSumET_BXm1Filled;

  MonitorElement* meMETPhiChargedHadronsBarrel_BXm1Filled;
  MonitorElement* meMETPhiChargedHadronsEndcapPlus_BXm1Filled;
  MonitorElement* meMETPhiChargedHadronsEndcapMinus_BXm1Filled;
  MonitorElement* meMETPhiNeutralHadronsBarrel_BXm1Filled;
  MonitorElement* meMETPhiNeutralHadronsEndcapPlus_BXm1Filled;
  MonitorElement* meMETPhiNeutralHadronsEndcapMinus_BXm1Filled;
  MonitorElement* meMETPhiPhotonsBarrel_BXm1Filled;
  MonitorElement* meMETPhiPhotonsEndcapPlus_BXm1Filled;
  MonitorElement* meMETPhiPhotonsEndcapMinus_BXm1Filled;
  MonitorElement* meMETPhiHFHadronsPlus_BXm1Filled;
  MonitorElement* meMETPhiHFHadronsMinus_BXm1Filled;
  MonitorElement* meMETPhiHFEGammasPlus_BXm1Filled;
  MonitorElement* meMETPhiHFEGammasMinus_BXm1Filled;

  //MEs where we fill if two previous bunches are filled (25 ns bunch spacing)
  MonitorElement* meChargedHadronEtFraction_BXm2BXm1Filled;
  MonitorElement* mePhotonEtFraction_BXm2BXm1Filled;
  MonitorElement* meNeutralHadronEtFraction_BXm2BXm1Filled;
  MonitorElement* meMET_BXm2BXm1Filled;
  MonitorElement* meSumET_BXm2BXm1Filled;

  MonitorElement* meCHF_Barrel;
  MonitorElement* meCHF_EndcapPlus;
  MonitorElement* meCHF_EndcapMinus;
  MonitorElement* meCHF_Barrel_BXm1Empty;
  MonitorElement* meCHF_EndcapPlus_BXm1Empty;
  MonitorElement* meCHF_EndcapMinus_BXm1Empty;
  MonitorElement* meCHF_Barrel_BXm2BXm1Empty;
  MonitorElement* meCHF_EndcapPlus_BXm2BXm1Empty;
  MonitorElement* meCHF_EndcapMinus_BXm2BXm1Empty;
  MonitorElement* meCHF_Barrel_BXm1Filled;
  MonitorElement* meCHF_EndcapPlus_BXm1Filled;
  MonitorElement* meCHF_EndcapMinus_BXm1Filled;
  MonitorElement* meCHF_Barrel_BXm2BXm1Filled;
  MonitorElement* meCHF_EndcapPlus_BXm2BXm1Filled;
  MonitorElement* meCHF_EndcapMinus_BXm2BXm1Filled;

  MonitorElement* meNHF_Barrel;
  MonitorElement* meNHF_EndcapPlus;
  MonitorElement* meNHF_EndcapMinus;
  MonitorElement* meNHF_Barrel_BXm1Empty;
  MonitorElement* meNHF_EndcapPlus_BXm1Empty;
  MonitorElement* meNHF_EndcapMinus_BXm1Empty;
  MonitorElement* meNHF_Barrel_BXm2BXm1Empty;
  MonitorElement* meNHF_EndcapPlus_BXm2BXm1Empty;
  MonitorElement* meNHF_EndcapMinus_BXm2BXm1Empty;
  MonitorElement* meNHF_Barrel_BXm1Filled;
  MonitorElement* meNHF_EndcapPlus_BXm1Filled;
  MonitorElement* meNHF_EndcapMinus_BXm1Filled;
  MonitorElement* meNHF_Barrel_BXm2BXm1Filled;
  MonitorElement* meNHF_EndcapPlus_BXm2BXm1Filled;
  MonitorElement* meNHF_EndcapMinus_BXm2BXm1Filled;

  MonitorElement* mePhF_Barrel;
  MonitorElement* mePhF_EndcapPlus;
  MonitorElement* mePhF_EndcapMinus;
  MonitorElement* mePhF_Barrel_BXm1Empty;
  MonitorElement* mePhF_EndcapPlus_BXm1Empty;
  MonitorElement* mePhF_EndcapMinus_BXm1Empty;
  MonitorElement* mePhF_Barrel_BXm2BXm1Empty;
  MonitorElement* mePhF_EndcapPlus_BXm2BXm1Empty;
  MonitorElement* mePhF_EndcapMinus_BXm2BXm1Empty;
  MonitorElement* mePhF_Barrel_BXm1Filled;
  MonitorElement* mePhF_EndcapPlus_BXm1Filled;
  MonitorElement* mePhF_EndcapMinus_BXm1Filled;
  MonitorElement* mePhF_Barrel_BXm2BXm1Filled;
  MonitorElement* mePhF_EndcapPlus_BXm2BXm1Filled;
  MonitorElement* mePhF_EndcapMinus_BXm2BXm1Filled;

  MonitorElement* meHFHadF_Plus;
  MonitorElement* meHFHadF_Minus;
  MonitorElement* meHFHadF_Plus_BXm1Empty;
  MonitorElement* meHFHadF_Minus_BXm1Empty;
  MonitorElement* meHFHadF_Plus_BXm2BXm1Empty;
  MonitorElement* meHFHadF_Minus_BXm2BXm1Empty;
  MonitorElement* meHFHadF_Plus_BXm1Filled;
  MonitorElement* meHFHadF_Minus_BXm1Filled;
  MonitorElement* meHFHadF_Plus_BXm2BXm1Filled;
  MonitorElement* meHFHadF_Minus_BXm2BXm1Filled;

  MonitorElement* meHFEMF_Plus;
  MonitorElement* meHFEMF_Minus;
  MonitorElement* meHFEMF_Plus_BXm1Empty;
  MonitorElement* meHFEMF_Minus_BXm1Empty;
  MonitorElement* meHFEMF_Plus_BXm2BXm1Empty;
  MonitorElement* meHFEMF_Minus_BXm2BXm1Empty;
  MonitorElement* meHFEMF_Plus_BXm1Filled;
  MonitorElement* meHFEMF_Minus_BXm1Filled;
  MonitorElement* meHFEMF_Plus_BXm2BXm1Filled;
  MonitorElement* meHFEMF_Minus_BXm2BXm1Filled;

  MonitorElement* meMETPhiChargedHadronsBarrel_BXm2BXm1Filled;
  MonitorElement* meMETPhiChargedHadronsEndcapPlus_BXm2BXm1Filled;
  MonitorElement* meMETPhiChargedHadronsEndcapMinus_BXm2BXm1Filled;
  MonitorElement* meMETPhiNeutralHadronsBarrel_BXm2BXm1Filled;
  MonitorElement* meMETPhiNeutralHadronsEndcapPlus_BXm2BXm1Filled;
  MonitorElement* meMETPhiNeutralHadronsEndcapMinus_BXm2BXm1Filled;
  MonitorElement* meMETPhiPhotonsBarrel_BXm2BXm1Filled;
  MonitorElement* meMETPhiPhotonsEndcapPlus_BXm2BXm1Filled;
  MonitorElement* meMETPhiPhotonsEndcapMinus_BXm2BXm1Filled;
  MonitorElement* meMETPhiHFHadronsPlus_BXm2BXm1Filled;
  MonitorElement* meMETPhiHFHadronsMinus_BXm2BXm1Filled;
  MonitorElement* meMETPhiHFEGammasPlus_BXm2BXm1Filled;
  MonitorElement* meMETPhiHFEGammasMinus_BXm2BXm1Filled;

  double ptMinCand_;

  // Smallest raw HCAL energy linked to the track
  double hcalMin_;
  MonitorElement* mProfileIsoPFChHad_HadPtCentral;
  MonitorElement* mProfileIsoPFChHad_HadPtEndcap;
  MonitorElement* mProfileIsoPFChHad_EMPtCentral;
  MonitorElement* mProfileIsoPFChHad_EMPtEndcap;
  MonitorElement* mProfileIsoPFChHad_TrackPt;

  MonitorElement* mProfileIsoPFChHad_HcalOccupancyCentral;
  MonitorElement* mProfileIsoPFChHad_HcalOccupancyEndcap;
  MonitorElement* mProfileIsoPFChHad_EcalOccupancyCentral;
  MonitorElement* mProfileIsoPFChHad_EcalOccupancyEndcap;
  MonitorElement* mProfileIsoPFChHad_TrackOccupancy;

  //PFcandidate maps
  std::vector<MonitorElement* > profilePFCand_x_,profilePFCand_y_,occupancyPFCand_,ptPFCand_,multiplicityPFCand_;
  std::vector<std::string> profilePFCand_x_name_,profilePFCand_y_name_,occupancyPFCand_name_,ptPFCand_name_,multiplicityPFCand_name_;
  std::vector<double> etaMinPFCand_, etaMaxPFCand_, MExPFCand_, MEyPFCand_;
  std::vector<int> typePFCand_, nbinsPFCand_, countsPFCand_, etaNBinsPFCand_;

  MonitorElement* meMETPhiChargedHadronsBarrel;
  MonitorElement* meMETPhiChargedHadronsEndcapPlus;
  MonitorElement* meMETPhiChargedHadronsEndcapMinus;
  MonitorElement* meMETPhiNeutralHadronsBarrel;
  MonitorElement* meMETPhiNeutralHadronsEndcapPlus;
  MonitorElement* meMETPhiNeutralHadronsEndcapMinus;
  MonitorElement* meMETPhiPhotonsBarrel;
  MonitorElement* meMETPhiPhotonsEndcapPlus;
  MonitorElement* meMETPhiPhotonsEndcapMinus;
  MonitorElement* meMETPhiHFHadronsPlus;
  MonitorElement* meMETPhiHFHadronsMinus;
  MonitorElement* meMETPhiHFEGammasPlus;
  MonitorElement* meMETPhiHFEGammasMinus;
 
  edm::EDGetTokenT<std::vector<reco::PFCandidate> > pflowToken_;

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
  MonitorElement* meChargedHadronEtFraction_profile;
  MonitorElement* meChargedHadronEt_profile;
  MonitorElement* meHFHadronEtFraction_profile;
  MonitorElement* meHFHadronEt_profile;
  MonitorElement* meHFEMEtFraction_profile;
  MonitorElement* meHFEMEt_profile;

  MonitorElement* meZJets_u_par;
  MonitorElement* meZJets_u_par_ZPt_0_15;
  MonitorElement* meZJets_u_par_ZPt_15_30;
  MonitorElement* meZJets_u_par_ZPt_30_55;
  MonitorElement* meZJets_u_par_ZPt_55_75;
  MonitorElement* meZJets_u_par_ZPt_75_150;
  MonitorElement* meZJets_u_par_ZPt_150_290;
  MonitorElement* meZJets_u_par_ZPt_290;
  
  MonitorElement* meZJets_u_perp;
  MonitorElement* meZJets_u_perp_ZPt_0_15;
  MonitorElement* meZJets_u_perp_ZPt_15_30;
  MonitorElement* meZJets_u_perp_ZPt_30_55;
  MonitorElement* meZJets_u_perp_ZPt_55_75;
  MonitorElement* meZJets_u_perp_ZPt_75_150;
  MonitorElement* meZJets_u_perp_ZPt_150_290;
  MonitorElement* meZJets_u_perp_ZPt_290;

  std::map< std::string,MonitorElement* >map_dijet_MEs;
  std::vector<unsigned int> nCh;
  std::vector<unsigned int> nEv;

  bool isCaloMet_;
  bool isPFMet_;
  bool isMiniAODMet_;

  bool fill_met_high_level_histo;
  bool fillCandidateMap_histos;

};
#endif
