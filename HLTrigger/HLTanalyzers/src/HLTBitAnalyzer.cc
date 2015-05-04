// File: HLTBitAnalyzer.cc
// Description:  Example of Analysis driver originally from Jeremy Mans, 
// Date:  13-October-2006

#include <boost/foreach.hpp>

#include "HLTrigger/HLTanalyzers/interface/HLTBitAnalyzer.h"
#include "HLTMessages.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

typedef std::pair<const char *, const edm::InputTag *> MissingCollectionInfo;
  
template <class T>
static inline
bool getCollection(const edm::Event & event, std::vector<MissingCollectionInfo> & missing, edm::Handle<T> & handle, const edm::InputTag & name, const edm::EDGetTokenT<T> token, const char * description) 
{
  event.getByToken(token, handle);
  bool valid = handle.isValid();
  if (not valid) {
    missing.push_back( std::make_pair(description, & name) );
    handle.clear();
    //	std::cout << "not valid "<< description << " " << name << std::endl;
  }
  return valid;
}

// Boiler-plate constructor definition of an analyzer module:
HLTBitAnalyzer::HLTBitAnalyzer(edm::ParameterSet const& conf)  :
  hlt_analysis_(conf, consumesCollector(), *this) {

  // If your module takes parameters, here is where you would define
  // their names and types, and access them to initialize internal
  // variables. Example as follows:
  std::cout << " Beginning HLTBitAnalyzer Analysis " << std::endl;

  l1extramu_        = conf.getParameter<std::string>   ("l1extramu");
  m_l1extramu       = edm::InputTag(l1extramu_, "");

  // read the L1Extra collection name, and add the instance names as needed
  l1extramc_        = conf.getParameter<std::string>   ("l1extramc");
  m_l1extraemi      = edm::InputTag(l1extramc_, "Isolated");
  m_l1extraemn      = edm::InputTag(l1extramc_, "NonIsolated");
  m_l1extrajetc     = edm::InputTag(l1extramc_, "Central");
  m_l1extrajetf     = edm::InputTag(l1extramc_, "Forward");
  m_l1extrajet      = edm::InputTag("gctInternJetProducer","Internal","ANALYSIS");
  m_l1extrataujet   = edm::InputTag(l1extramc_, "Tau");
  m_l1extramet      = edm::InputTag(l1extramc_, "MET");
  m_l1extramht      = edm::InputTag(l1extramc_, "MHT");

  hltresults_       = conf.getParameter<edm::InputTag> ("hltresults");
  gtReadoutRecord_  = conf.getParameter<edm::InputTag> ("l1GtReadoutRecord");
  gtObjectMap_      = conf.getParameter<edm::InputTag> ("l1GtObjectMapRecord");

  gctBitCounts_        = edm::InputTag( conf.getParameter<edm::InputTag>("l1GctHFBitCounts").label(), "" );
  gctRingSums_         = edm::InputTag( conf.getParameter<edm::InputTag>("l1GctHFRingSums").label(), "" );

  hltresultsToken_ = consumes<edm::TriggerResults>(hltresults_);
  genEventInfoToken_ = consumes<GenEventInfoProduct>(genEventInfo_);
  l1extramuToken_ = consumes<l1extra::L1MuonParticleCollection>(m_l1extramu);
  l1extraemiToken_ = consumes<l1extra::L1EmParticleCollection>(m_l1extraemi);
  l1extraemnToken_ = consumes<l1extra::L1EmParticleCollection>(m_l1extraemn);
  
  l1extrajetcToken_ = consumes<l1extra::L1JetParticleCollection>(m_l1extrajetc);
  l1extrajetfToken_ = consumes<l1extra::L1JetParticleCollection>(m_l1extrajetf);
  l1extrajetToken_ = consumes<l1extra::L1JetParticleCollection>(m_l1extrajet);
  l1extrataujetToken_ = consumes<l1extra::L1JetParticleCollection>(m_l1extrataujet);
  l1extrametToken_ = consumes<l1extra::L1EtMissParticleCollection>(m_l1extramet);
  l1extramhtToken_ = consumes<l1extra::L1EtMissParticleCollection>(m_l1extramht);
  gtReadoutRecordToken_ = consumes<L1GlobalTriggerReadoutRecord>(gtReadoutRecord_);
  gtObjectMapToken_ = consumes<L1GlobalTriggerObjectMapRecord>(gtObjectMap_);
  gctBitCountsToken_ = consumes<L1GctHFBitCountsCollection>(gctBitCounts_);
  gctRingSumsToken_ = consumes<L1GctHFRingEtSumsCollection>(gctRingSums_);
  
  _UseTFileService = conf.getUntrackedParameter<bool>("UseTFileService",false);
	
  m_file = 0;   // set to null
  errCnt = 0;

  // read run parameters with a default value
  edm::ParameterSet runParameters = conf.getParameter<edm::ParameterSet>("RunParameters");
  _HistName = runParameters.getUntrackedParameter<std::string>("HistogramFile", "test.root");

  // open the tree file and initialize the tree
  if(_UseTFileService){
    edm::Service<TFileService> fs;
    HltTree = fs->make<TTree>("HltTree", "");
  }else{
    m_file = new TFile(_HistName.c_str(), "RECREATE");
    if (m_file)
      m_file->cd();
    HltTree = new TTree("HltTree", "");
  }

  // Setup the different analysis
  hlt_analysis_.setup(conf, HltTree);
  evt_header_.setup(consumesCollector(), HltTree);
}

// Boiler-plate "analyze" method declaration for an analyzer module.
void HLTBitAnalyzer::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {

  edm::Handle<edm::TriggerResults>                  hltresults;
  edm::Handle<l1extra::L1EmParticleCollection>      l1extemi, l1extemn;
  edm::Handle<l1extra::L1MuonParticleCollection>    l1extmu;
  edm::Handle<l1extra::L1JetParticleCollection>     l1extjetc, l1extjetf, l1extjet, l1exttaujet;
  edm::Handle<l1extra::L1EtMissParticleCollection>  l1extmet,l1extmht;
  edm::Handle<L1GlobalTriggerReadoutRecord>         l1GtRR;
  edm::Handle<L1GlobalTriggerObjectMapRecord>       l1GtOMRec;
  edm::Handle<L1GlobalTriggerObjectMap>             l1GtOM;
  edm::Handle< L1GctHFBitCountsCollection >         gctBitCounts ;
  edm::Handle< L1GctHFRingEtSumsCollection >        gctRingSums ;

  // extract the collections from the event, check their validity and log which are missing
  std::vector<MissingCollectionInfo> missing;

  getCollection( iEvent, missing, hltresults,      hltresults_,        hltresultsToken_,      kHltresults );
  getCollection( iEvent, missing, l1extemi,        m_l1extraemi,       l1extraemiToken_,      kL1extemi );
  getCollection( iEvent, missing, l1extemn,        m_l1extraemn,       l1extraemnToken_,      kL1extemn );
  getCollection( iEvent, missing, l1extmu,         m_l1extramu,        l1extramuToken_,       kL1extmu );
  getCollection( iEvent, missing, l1extjetc,       m_l1extrajetc,      l1extrajetcToken_,     kL1extjetc );
  getCollection( iEvent, missing, l1extjetf,       m_l1extrajetf,      l1extrajetfToken_,     kL1extjetf );
  getCollection( iEvent, missing, l1extjet,        m_l1extrajet,       l1extrajetToken_,      kL1extjet );
  getCollection( iEvent, missing, l1exttaujet,     m_l1extrataujet,    l1extrataujetToken_,   kL1exttaujet );
  getCollection( iEvent, missing, l1extmet,        m_l1extramet,       l1extrametToken_,      kL1extmet );
  getCollection( iEvent, missing, l1extmht,        m_l1extramht,       l1extramhtToken_,      kL1extmht );
  getCollection( iEvent, missing, l1GtRR,          gtReadoutRecord_,   gtReadoutRecordToken_, kL1GtRR );
  getCollection( iEvent, missing, l1GtOMRec,       gtObjectMap_,       gtObjectMapToken_,     kL1GtOMRec );
  getCollection( iEvent, missing, gctBitCounts,    gctBitCounts_,      gctBitCountsToken_,    kL1GctBitCounts );
  getCollection( iEvent, missing, gctRingSums,     gctRingSums_,       gctRingSumsToken_,     kL1GctRingSums );


  // print missing collections
  if (not missing.empty() and (errCnt < errMax())) {
    errCnt++;
    std::stringstream out;       
    out <<  "OpenHLT analyser - missing collections:";
    BOOST_FOREACH(const MissingCollectionInfo & entry, missing)
      out << "\n\t" << entry.first << ": " << entry.second->encode();
    edm::LogPrint("OpenHLT") << out.str() << std::endl; 
    if (errCnt == errMax())
      edm::LogWarning("OpenHLT") << "Maximum error count reached -- No more messages will be printed.";
  }

  // run the analysis, passing required event fragments
  hlt_analysis_.analyze(
    hltresults,
    l1extemi,
    l1extemn,
    l1extmu,
    l1extjetc,
    l1extjetf,
    l1extjet,
    l1exttaujet,
    l1extmet,
    l1extmht,
    l1GtRR,
    gctBitCounts,
    gctRingSums,
    iSetup,
    iEvent,
    HltTree);

  evt_header_.analyze(iEvent, HltTree);

  // std::cout << " Ending Event Analysis" << std::endl;
  // After analysis, fill the variables tree
  if (m_file)
    m_file->cd();
  HltTree->Fill();
}

// ------------ method called when starting to processes a run  ------------
void 
HLTBitAnalyzer::beginRun(edm::Run const& run, edm::EventSetup const& es)
{
   hlt_analysis_.beginRun(run, es);
}

// "endJob" is an inherited method that you may implement to do post-EOF processing and produce final output.
void HLTBitAnalyzer::endJob() {
	
	if(!_UseTFileService){
		if (m_file)
			m_file->cd();
		
		HltTree->Write();
		delete HltTree;
		HltTree = 0;
		
		if (m_file) {         // if there was a tree file...
			m_file->Write();    // write out the branches
			delete m_file;      // close and delete the file
			m_file = 0;         // set to zero to clean up
		}
	}
}
