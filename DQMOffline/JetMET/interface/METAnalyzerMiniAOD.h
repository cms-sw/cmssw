#ifndef METAnalyzerMiniAOD_H
#define METAnalyzerMiniAOD_H

/** \class JetMETAnalyzerMiniAOD
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
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"


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

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/METCollection.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DQMOffline/JetMET/interface/JetMETDQMDCSFilter.h"
#include "CommonTools/RecoAlgos/interface/HBHENoiseFilter.h"
#include "PhysicsTools/SelectorUtils/interface/JetIDSelectionFunctor.h"
#include "PhysicsTools/SelectorUtils/interface/PFJetIDSelectionFunctor.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include <map>
#include <string>



class METAnalyzerMiniAOD : public thread_unsafe::DQMEDAnalyzer{
 public:

  /// Constructor
  METAnalyzerMiniAOD(const edm::ParameterSet&);

  /// Destructor
  virtual ~METAnalyzerMiniAOD();

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
  void fillMESet(const edm::Event&, std::string, const pat::MET&,std::map<std::string,MonitorElement*>&);
  void fillMonitorElement(const edm::Event&, std::string, std::string, const pat::MET&, std::map<std::string,MonitorElement*>&,bool);
  void makeRatePlot(std::string, double);

//  bool selectHighPtJetEvent(const edm::Event&);
//  bool selectLowPtJetEvent(const edm::Event&);
//  bool selectWElectronEvent(const edm::Event&);
//  bool selectWMuonEvent(const edm::Event&);

 private:

 // Book MonitorElements
  void bookMESet(std::string,DQMStore::IBooker &,std::map<std::string,MonitorElement*>&);
// Book MonitorElements
  void bookMonitorElement(std::string,DQMStore::IBooker &, std::map<std::string,MonitorElement*>&,bool );

  // ----------member data ---------------------------
  edm::ParameterSet parameters;
  // Switch for verbosity
  int verbose_;


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
  edm::EDGetTokenT< edm::View<pat::Jet> >        patJetsToken_;

  edm::EDGetTokenT<bool>                          hbheNoiseFilterResultToken_;
  edm::EDGetTokenT<reco::BeamHaloSummary>         beamHaloSummaryToken_;
  edm::EDGetTokenT<pat::METCollection>        patMETToken_;
  edm::EDGetTokenT<std::vector<pat::PackedCandidate> > patCandToken_;

  JetIDSelectionFunctor jetIDFunctorLoose;
  PFJetIDSelectionFunctor pfjetIDFunctorLoose;

  double ptThreshold_;

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

  //now PF only things
  MonitorElement* mePhotonEtFraction;
  MonitorElement* meNeutralHadronEtFraction;
  MonitorElement* meElectronEtFraction;
  MonitorElement* meChargedHadronEtFraction;
  MonitorElement* meMuonEtFraction;
  MonitorElement* meHFHadronEtFraction;
  MonitorElement* meHFEMEtFraction;
 

  // NPV profiles --> 
  //----------------------------------------------------------------------------
  MonitorElement* meMEx_profile;
  MonitorElement* meMEy_profile;
  MonitorElement* meMET_profile;
  MonitorElement* meSumET_profile;

  MonitorElement* mePhotonEtFraction_profile;
  MonitorElement* meNeutralHadronEtFraction_profile;
  MonitorElement* meElectronEtFraction_profile;
  MonitorElement* meChargedHadronEtFraction_profile;
  MonitorElement* meMuonEtFraction_profile;
  MonitorElement* meHFHadronEtFraction_profile;
  MonitorElement* meHFEMEtFraction_profile;

  //PFcandidate maps
  std::vector<MonitorElement* > profilePFCand_x_,profilePFCand_y_,occupancyPFCand_,energyPFCand_,ptPFCand_,multiplicityPFCand_;
  std::vector<std::string> profilePFCand_x_name_,profilePFCand_y_name_,occupancyPFCand_name_,energyPFCand_name_,ptPFCand_name_,multiplicityPFCand_name_;
  std::vector<double> etaMinPFCand_, etaMaxPFCand_, MExPFCand_, MEyPFCand_;
  std::vector<int> typePFCand_, nbinsPFCand_, countsPFCand_, etaNBinsPFCand_;

  std::map< std::string,MonitorElement* >map_dijet_MEs;

  bool isCaloMet_;
  //bool isTCMet_;
  bool isPFMet_;

  bool fill_met_high_level_histo;

};
#endif
