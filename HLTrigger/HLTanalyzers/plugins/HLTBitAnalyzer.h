#include <iostream>

#include "CondFormats/DataRecord/interface/L1TUtmTriggerMenuRcd.h"
#include "CondFormats/L1TObjects/interface/L1TUtmAlgorithm.h"
#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

#include "EventHeader.h"
#include "HLTInfo.h"
#include "HLTMCtruth.h"
#include "RECOVertex.h"

/** \class HLTBitAnalyzer
  *  
  * $Date: November 2006
  * $Revision: 
  * \author P. Bargassa - Rice U.

  * $Date: April 2016 
  * $Revision:   
  * \author G. Karapostoli - ULB
  */

class HLTBitAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit HLTBitAnalyzer(edm::ParameterSet const& conf);
  void analyze(edm::Event const& e, edm::EventSetup const& iSetup) override;
  void endJob() override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override {}

  //  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

  // Analysis tree to be filled
  TTree* HltTree;

private:
  // variables persistent across events should be declared here.
  //
  ///Default analyses

  EventHeader evt_header_;
  HLTInfo hlt_analysis_;

  HLTMCtruth mct_analysis_;
  RECOVertex vrt_analysisOffline0_;

  edm::InputTag hltresults_, genEventInfo_;
  /*
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
  edm::InputTag gtReadoutRecord_,gtObjectMap_; 
  edm::InputTag gctBitCounts_,gctRingSums_;
  */
  edm::InputTag l1results_;

  edm::InputTag mctruth_, simhits_;
  edm::InputTag VertexTagOffline0_;
  edm::InputTag pileupInfo_;

  edm::EDGetTokenT<edm::TriggerResults> hltresultsToken_;
  edm::EDGetTokenT<GenEventInfoProduct> genEventInfoToken_;

  edm::EDGetTokenT<GlobalAlgBlkBxCollection> l1resultsToken_;
  /*
  edm::EDGetTokenT<l1extra::L1MuonParticleCollection>    l1extramuToken_;
  edm::EDGetTokenT<l1extra::L1EmParticleCollection>      l1extraemiToken_, l1extraemnToken_;
  edm::EDGetTokenT<l1extra::L1JetParticleCollection>     l1extrajetcToken_, l1extrajetfToken_, l1extrajetToken_, l1extrataujetToken_;
  edm::EDGetTokenT<l1extra::L1EtMissParticleCollection>  l1extrametToken_,l1extramhtToken_;

  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord>         gtReadoutRecordToken_;
  edm::EDGetTokenT<L1GlobalTriggerObjectMapRecord>       gtObjectMapToken_;
  edm::EDGetTokenT< L1GctHFBitCountsCollection >         gctBitCountsToken_;
  edm::EDGetTokenT< L1GctHFRingEtSumsCollection >        gctRingSumsToken_;
  */
  edm::EDGetTokenT<reco::CandidateView> mctruthToken_;
  edm::EDGetTokenT<std::vector<SimTrack> > simtracksToken_;
  edm::EDGetTokenT<std::vector<SimVertex> > simverticesToken_;
  edm::EDGetTokenT<std::vector<PileupSummaryInfo> > pileupInfoToken_;
  edm::EDGetTokenT<reco::VertexCollection> VertexTagOffline0Token_;

  int errCnt;
  static int errMax() { return 5; }

  std::string _HistName;  // Name of histogram file
  double _EtaMin, _EtaMax;
  TFile* m_file;  // pointer to Histogram file
  bool _UseTFileService;
  bool _isData;

  double ptHat, weight;
};
