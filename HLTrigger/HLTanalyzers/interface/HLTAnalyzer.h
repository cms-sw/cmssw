#include <iostream>

#include "HLTrigger/HLTanalyzers/interface/HLTEgamma.h"
#include "HLTrigger/HLTanalyzers/interface/HLTInfo.h"
#include "HLTrigger/HLTanalyzers/interface/HLTJets.h"
#include "HLTrigger/HLTanalyzers/interface/HLTBJet.h"
#include "HLTrigger/HLTanalyzers/interface/HLTMCtruth.h"
#include "HLTrigger/HLTanalyzers/interface/HLTMuon.h"
#include "HLTrigger/HLTanalyzers/interface/HLTAlCa.h"  
#include "HLTrigger/HLTanalyzers/interface/HLTTrack.h"
#include "HLTrigger/HLTanalyzers/interface/EventHeader.h"
#include "HLTrigger/HLTanalyzers/interface/RECOVertex.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/Registry.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"  

#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h"  

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

/** \class HLTAnalyzer
  *  
  * $Date: November 2006
  * $Revision: 
  * \author P. Bargassa - Rice U.
  */

class HLTAnalyzer : public edm::EDAnalyzer {
public:
  explicit HLTAnalyzer(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
  virtual void beginRun(const edm::Run& , const edm::EventSetup& );
  virtual void endJob();

  //  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions); 

  // Analysis tree to be filled
  TTree *HltTree;

private:
  // variables persistent across events should be declared here.
  //
  ///Default analyses

  EventHeader evt_header_;
  HLTJets     jet_analysis_;
  HLTBJet     bjet_analysis_;
  HLTMuon     muon_analysis_;
  HLTEgamma   elm_analysis_;
  HLTMCtruth  mct_analysis_;
  /*
  HLTAlCa     alca_analysis_; 
  */
  HLTTrack    track_analysis_;
  HLTInfo     hlt_analysis_;
  RECOVertex  vrt_analysisHLT_;
  RECOVertex  vrt_analysisOffline0_;

  int firstLumi_, lastLumi_, towerThreshold_;
  double xSection_, filterEff_, treeWeight;

  //
  // All tokens needed to access products in the event
  //

  edm::EDGetTokenT<reco::BeamSpot>                       BSProducerToken_;
  edm::EDGetTokenT<reco::CaloJetCollection>              hltjetsToken_;
  edm::EDGetTokenT<reco::CaloJetCollection>              hltcorjetsToken_;
  edm::EDGetTokenT<reco::CaloJetCollection>              hltcorL1L2L3jetsToken_;
  edm::EDGetTokenT<double>                               rhoToken_;
  edm::EDGetTokenT<reco::CaloJetCollection>              recjetsToken_;
  edm::EDGetTokenT<reco::CaloJetCollection>              reccorjetsToken_;
  edm::EDGetTokenT<reco::GenJetCollection>               genjetsToken_;
  edm::EDGetTokenT<CaloTowerCollection>                  calotowersToken_;
  edm::EDGetTokenT<CaloTowerCollection>                  calotowersUpperR45Token_;
  edm::EDGetTokenT<CaloTowerCollection>                  calotowersLowerR45Token_;
  edm::EDGetTokenT<CaloTowerCollection>                  calotowersNoR45Token_;
  edm::EDGetTokenT<reco::CaloMETCollection>              recmetToken_;
  edm::EDGetTokenT<reco::PFMETCollection>                recoPFMetToken_;
  edm::EDGetTokenT<reco::GenMETCollection>               genmetToken_;
  edm::EDGetTokenT<reco::METCollection>                  htToken_;
  edm::EDGetTokenT<reco::PFJetCollection>                recoPFJetsToken_; 
  edm::EDGetTokenT<reco::CandidateView>                  mctruthToken_;
  edm::EDGetTokenT<GenEventInfoProduct>                  genEventInfoToken_;
  edm::EDGetTokenT<std::vector<SimTrack> >               simTracksToken_;
  edm::EDGetTokenT<std::vector<SimVertex> >              simVerticesToken_;
  edm::EDGetTokenT<reco::MuonCollection>                 muonToken_;
  edm::EDGetTokenT<reco::PFCandidateCollection>          pfmuonToken_;
  edm::EDGetTokenT<edm::TriggerResults>                  hltresultsToken_;
  edm::EDGetTokenT<l1extra::L1EmParticleCollection>      l1extraemiToken_, l1extraemnToken_;
  edm::EDGetTokenT<l1extra::L1MuonParticleCollection>    l1extramuToken_;
  edm::EDGetTokenT<l1extra::L1JetParticleCollection>     l1extrajetcToken_, l1extrajetfToken_, l1extrajetToken_, l1extrataujetToken_;
  edm::EDGetTokenT<l1extra::L1EtMissParticleCollection>  l1extrametToken_,l1extramhtToken_;
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord>         gtReadoutRecordToken_;
  edm::EDGetTokenT< L1GctHFBitCountsCollection >         gctBitCountsToken_;
  edm::EDGetTokenT< L1GctHFRingEtSumsCollection >        gctRingSumsToken_;
    
  edm::EDGetTokenT<reco::RecoChargedCandidateCollection> MuCandTag2Token_, MuCandTag3Token_, MuNoVtxCandTag2Token_;
  edm::EDGetTokenT<reco::RecoChargedCandidateCollection> oniaPixelTagToken_, oniaTrackTagToken_;
  edm::EDGetTokenT<reco::VertexCollection>               DiMuVtxToken_;
  edm::EDGetTokenT<reco::MuonCollection>                 TrackerMuonTagToken_;
  edm::EDGetTokenT<edm::ValueMap<bool> >                 MuIsolTag2Token_,  MuIsolTag3Token_, MuTrkIsolTag3Token_;
  edm::EDGetTokenT<reco::CaloJetCollection>              L2TauToken_;
  edm::EDGetTokenT<reco::HLTTauCollection>               HLTTauToken_;
  edm::EDGetTokenT<reco::PFTauCollection>                PFTauToken_;
  edm::EDGetTokenT<reco::PFTauCollection>                PFTauTightConeToken_;
  edm::EDGetTokenT<reco::PFJetCollection>                PFJetsToken_;
    
    // offline reco tau collection and discriminators
  edm::EDGetTokenT<reco::PFTauCollection>    RecoPFTauToken_;
  edm::EDGetTokenT<reco::PFTauDiscriminator> RecoPFTauDiscrByTanCOnePercentToken_;
  edm::EDGetTokenT<reco::PFTauDiscriminator> RecoPFTauDiscrByTanCHalfPercentToken_; 
  edm::EDGetTokenT<reco::PFTauDiscriminator> RecoPFTauDiscrByTanCQuarterPercentToken_;
  edm::EDGetTokenT<reco::PFTauDiscriminator> RecoPFTauDiscrByTanCTenthPercentToken_;
  edm::EDGetTokenT<reco::PFTauDiscriminator> RecoPFTauDiscrByIsoToken_;
  edm::EDGetTokenT<reco::PFTauDiscriminator> RecoPFTauAgainstMuonToken_;
  edm::EDGetTokenT<reco::PFTauDiscriminator> RecoPFTauAgainstElecToken_;
       
    // btag OpenHLT input collections
  edm::EDGetTokenT<edm::View<reco::Jet> >                rawBJetsToken_;
  edm::EDGetTokenT<edm::View<reco::Jet> >                correctedBJetsToken_;
  edm::EDGetTokenT<edm::View<reco::Jet> >                correctedBJetsL1FastJetToken_;
  edm::EDGetTokenT<edm::View<reco::Jet> >                pfBJetsToken_;
  edm::EDGetTokenT<reco::JetTagCollection>               lifetimeBJetsL25Token_;
  edm::EDGetTokenT<reco::JetTagCollection>               lifetimeBJetsL3L1FastJetToken_;
  edm::EDGetTokenT<reco::JetTagCollection>               lifetimeBJetsL25L1FastJetToken_;
  edm::EDGetTokenT<reco::JetTagCollection>               lifetimeBJetsL3Token_;
  edm::EDGetTokenT<reco::JetTagCollection>               lifetimePFBJetsL3Token_;
  edm::EDGetTokenT<reco::JetTagCollection>               lifetimeBJetsL25SingleTrackToken_;
  edm::EDGetTokenT<reco::JetTagCollection>               lifetimeBJetsL3SingleTrackToken_;
  edm::EDGetTokenT<reco::JetTagCollection>               lifetimeBJetsL25SingleTrackL1FastJetToken_;
  edm::EDGetTokenT<reco::JetTagCollection>               lifetimeBJetsL3SingleTrackL1FastJetToken_;
  edm::EDGetTokenT<reco::JetTagCollection>               performanceBJetsL25Token_;
  edm::EDGetTokenT<reco::JetTagCollection>               performanceBJetsL3Token_;
  edm::EDGetTokenT<reco::JetTagCollection>               performanceBJetsL25L1FastJetToken_;
  edm::EDGetTokenT<reco::JetTagCollection>               performanceBJetsL3L1FastJetToken_;
    
    // egamma OpenHLT input collections
  edm::EDGetTokenT<reco::GsfElectronCollection>          ElectronToken_;
  edm::EDGetTokenT<reco::PhotonCollection>               PhotonToken_;
  edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap>  IsoPhoR9Token_; 
  edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap>  NonIsoPhoR9Token_;
  edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap>  IsoPhoR9IDToken_;
  edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap>  NonIsoPhoR9IDToken_;
  edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap>  IsoPhoHoverEHToken_;   
  edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap>  NonIsoPhoHoverEHToken_;    
  edm::EDGetTokenT<reco::ElectronCollection>             IsoElectronToken_;
  edm::EDGetTokenT<reco::ElectronCollection>             NonIsoElectronToken_;
  edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap>  IsoEleR9Token_; 
  edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap>  NonIsoEleR9Token_;  
  edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap>  IsoEleR9IDToken_;
  edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap>  NonIsoEleR9IDToken_;
  edm::EDGetTokenT<reco::ElectronIsolationMap>           IsoEleTrackIsolToken_;
  edm::EDGetTokenT<reco::ElectronIsolationMap>           NonIsoEleTrackIsolToken_;
  edm::EDGetTokenT<reco::ElectronSeedCollection>         L1IsoPixelSeedsToken_;
  edm::EDGetTokenT<reco::ElectronSeedCollection>         L1NonIsoPixelSeedsToken_;
  edm::EDGetTokenT<reco::RecoEcalCandidateCollection>    CandIsoToken_;
  edm::EDGetTokenT<reco::RecoEcalCandidateCollection>    CandNonIsoToken_;
  edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap>  EcalIsoToken_;
  edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap>  EcalNonIsoToken_;
  edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap>  HcalIsoPhoToken_;
  edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap>  HcalNonIsoPhoToken_;
  edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap>  IsoEleHcalToken_;
  edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap>  NonIsoEleHcalToken_;
  edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap>  NonIsoPhoTrackIsolToken_;
  edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap>  IsoPhoTrackIsolToken_;
  edm::EDGetTokenT<reco::SuperClusterCollection>         HFECALClustersToken_;
  edm::EDGetTokenT<reco::RecoEcalCandidateCollection>    HFElectronsToken_;
  edm::EDGetTokenT<EcalRecHitCollection>                 EcalRecHitEBToken_;
  edm::EDGetTokenT<EcalRecHitCollection>                 EcalRecHitEEToken_;

    // ECAL Activity
  edm::EDGetTokenT<reco::RecoEcalCandidateCollection>    ECALActivityToken_;
  edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap>  ActivityEcalIsoToken_;
  edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap>  ActivityHcalIsoToken_;
  edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap>  ActivityTrackIsoToken_;
  edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap>  ActivityR9Token_;
  edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap>  ActivityR9IDToken_;
  edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap>  ActivityHoverEHToken_;
 
    
    // AlCa OpenHLT input collections
    /*
  edm::EDGetTokenT<EBRecHitCollection>             EBRecHitToken_;
  edm::EDGetTokenT<EERecHitCollection>             EERecHitToken_;
  edm::EDGetTokenT<EBRecHitCollection>             pi0EBRecHitToken_;
  edm::EDGetTokenT<EERecHitCollection>             pi0EERecHitToken_;
  edm::EDGetTokenT<HBHERecHitCollection>           HBHERecHitToken_;
  edm::EDGetTokenT<HORecHitCollection>             HORecHitToken_;
  edm::EDGetTokenT<HFRecHitCollection>             HFRecHitToken_;
    */

  edm::EDGetTokenT<reco::IsolatedPixelTrackCandidateCollection> IsoPixelTrackL3Token_; 
  edm::EDGetTokenT<reco::IsolatedPixelTrackCandidateCollection> IsoPixelTrackL2Token_;	
  edm::EDGetTokenT<reco::VertexCollection>                      IsoPixelTrackVerticesToken_;
  edm::EDGetTokenT<reco::RecoChargedCandidateCollection>        PixelTracksL3Token_; 
  edm::EDGetTokenT<FEDRawDataCollection>                        PixelFEDSizeToken_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> >       PixelClustersToken_;
    
    // Reco vertex collection
  edm::EDGetTokenT<reco::VertexCollection> VertexHLTToken_;
  edm::EDGetTokenT<reco::VertexCollection> VertexOffline0Token_;

    // Extra stuffs
  edm::EDGetTokenT<reco::HFEMClusterShapeAssociationCollection> HFEMClusterShapeAssociationToken_;

  //
  // All input tags
  //

  edm::InputTag BSProducer_;

  edm::InputTag recjets_,reccorjets_,genjets_,recmet_,recoPFMet_,genmet_,ht_,recoPFJets_,calotowers_,hltresults_,genEventInfo_;
  edm::InputTag calotowersUpperR45_, calotowersLowerR45_, calotowersNoR45_;
  edm::InputTag hltjets_, hltcorjets_, hltcorL1L2L3jets_, rho_;
  edm::InputTag muon_;
  edm::InputTag pfmuon_;
  std::string l1extramc_, l1extramu_;
  edm::InputTag m_l1extramu;
  edm::InputTag m_l1extraemi;
  edm::InputTag m_l1extraemn;
  edm::InputTag m_l1extrajetc;
  edm::InputTag m_l1extrajetf;
  edm::InputTag m_l1extrajet;
  edm::InputTag m_l1extrataujet;
  edm::InputTag m_l1extramet;
  edm::InputTag m_l1extramht;

  edm::InputTag particleMapSource_,mctruth_,simhits_; 
  edm::InputTag gtReadoutRecord_,gtObjectMap_; 
  edm::InputTag gctBitCounts_,gctRingSums_;

  edm::InputTag MuCandTag2_,MuIsolTag2_,MuNoVtxCandTag2_,MuCandTag3_,MuIsolTag3_,MuTrkIsolTag3_;
  edm::InputTag oniaPixelTag_,oniaTrackTag_,DiMuVtx_,TrackerMuonTag_;
  edm::InputTag L2Tau_, HLTTau_, PFTau_, PFTauTightCone_;
  edm::InputTag PFJets_;
  
  //offline reco tau collection and discriminators
  edm::InputTag RecoPFTau_;
  edm::InputTag RecoPFTauDiscrByTanCOnePercent_;
  edm::InputTag RecoPFTauDiscrByTanCHalfPercent_;
  edm::InputTag RecoPFTauDiscrByTanCQuarterPercent_;
  edm::InputTag RecoPFTauDiscrByTanCTenthPercent_;
  edm::InputTag RecoPFTauDiscrByIso_;
  edm::InputTag RecoPFTauAgainstMuon_;
  edm::InputTag RecoPFTauAgainstElec_;
  
 
  // btag OpenHLT input collections
  edm::InputTag m_rawBJets;
  edm::InputTag m_correctedBJets;
  edm::InputTag m_correctedBJetsL1FastJet;
  edm::InputTag m_pfBJets;
  edm::InputTag m_lifetimeBJetsL25;
  edm::InputTag m_lifetimeBJetsL3;
  edm::InputTag m_lifetimeBJetsL25L1FastJet;
  edm::InputTag m_lifetimeBJetsL3L1FastJet;
  edm::InputTag m_lifetimePFBJetsL3;
  edm::InputTag m_lifetimeBJetsL25SingleTrack;
  edm::InputTag m_lifetimeBJetsL3SingleTrack;
  edm::InputTag m_lifetimeBJetsL25SingleTrackL1FastJet;
  edm::InputTag m_lifetimeBJetsL3SingleTrackL1FastJet;
  edm::InputTag m_performanceBJetsL25;
  edm::InputTag m_performanceBJetsL3;
  edm::InputTag m_performanceBJetsL25L1FastJet;
  edm::InputTag m_performanceBJetsL3L1FastJet;

  // egamma OpenHLT input collections
  edm::InputTag Electron_;
  edm::InputTag Photon_;
  edm::InputTag CandIso_;
  edm::InputTag CandNonIso_;
  edm::InputTag EcalIso_;
  edm::InputTag EcalNonIso_;
  edm::InputTag HcalIsoPho_;
  edm::InputTag HcalNonIsoPho_;
  edm::InputTag IsoPhoTrackIsol_;
  edm::InputTag NonIsoPhoTrackIsol_;
  edm::InputTag IsoElectron_;
  edm::InputTag NonIsoElectron_;
  edm::InputTag IsoEleHcal_;
  edm::InputTag NonIsoEleHcal_;
  edm::InputTag IsoEleTrackIsol_;
  edm::InputTag NonIsoEleTrackIsol_;
  edm::InputTag L1IsoPixelSeeds_;
  edm::InputTag L1NonIsoPixelSeeds_;
  edm::InputTag NonIsoR9_; 
  edm::InputTag IsoR9_;  
  edm::InputTag NonIsoR9ID_;
  edm::InputTag IsoR9ID_;
  edm::InputTag IsoHoverEH_;
  edm::InputTag NonIsoHoverEH_; 
  edm::InputTag HFECALClusters_; 
  edm::InputTag HFElectrons_; 
  // add ECAL Activity
  edm::InputTag ECALActivity_;
  edm::InputTag ActivityEcalIso_;
  edm::InputTag ActivityHcalIso_;
  edm::InputTag ActivityTrackIso_;
  edm::InputTag ActivityR9_;
  edm::InputTag ActivityR9ID_;
  edm::InputTag ActivityHoverEH_;

  // AlCa OpenHLT input collections  
  /*
  edm::InputTag EERecHitTag_; 
  edm::InputTag EBRecHitTag_;  
  edm::InputTag pi0EERecHitTag_;  
  edm::InputTag pi0EBRecHitTag_;   
  edm::InputTag HBHERecHitTag_;   
  edm::InputTag HORecHitTag_;   
  edm::InputTag HFRecHitTag_;
  */   
  edm::InputTag IsoPixelTrackTagL3_;
  edm::InputTag IsoPixelTrackTagL2_; 
  edm::InputTag IsoPixelTrackVerticesTag_;


  // Track OpenHLT input collections

  edm::InputTag PixelTracksTagL3_; 
  edm::InputTag PixelFEDSizeTag_;
  edm::InputTag PixelClustersTag_;

  // Reco vertex collection
  edm::InputTag VertexTagHLT_;
  edm::InputTag VertexTagOffline0_;

  int errCnt;
  static int errMax() { return 5; }

  std::string _HistName; // Name of histogram file
  double _EtaMin,_EtaMax;
    double _MinPtChargedHadrons, _MinPtGammas;
  TFile* m_file; // pointer to Histogram file

};
