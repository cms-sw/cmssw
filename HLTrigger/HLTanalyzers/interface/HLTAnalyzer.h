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
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"  
#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"

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

  edm::InputTag pileupInfo_;

  int errCnt;
  static int errMax() { return 5; }

  std::string _HistName; // Name of histogram file
  double _EtaMin,_EtaMax;
    double _MinPtChargedHadrons, _MinPtGammas;
  TFile* m_file; // pointer to Histogram file

};
