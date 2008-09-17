// File: HLTAnalyzer.cc
// Description:  Example of Analysis driver originally from Jeremy Mans, 
// Date:  13-October-2006

#include <boost/foreach.hpp>

#include "HLTrigger/HLTanalyzers/interface/HLTAnalyzer.h"
#include "HLTMessages.h"

// Boiler-plate constructor definition of an analyzer module:
HLTAnalyzer::HLTAnalyzer(edm::ParameterSet const& conf) {

  // If your module takes parameters, here is where you would define
  // their names and types, and access them to initialize internal
  // variables. Example as follows:
  std::cout << " Beginning HLTAnalyzer Analysis " << std::endl;

  recjets_          = conf.getParameter<edm::InputTag> ("recjets");
  genjets_          = conf.getParameter<edm::InputTag> ("genjets");
  recmet_           = conf.getParameter<edm::InputTag> ("recmet");
  genmet_           = conf.getParameter<edm::InputTag> ("genmet");
  ht_               = conf.getParameter<edm::InputTag> ("ht");
  calotowers_       = conf.getParameter<edm::InputTag> ("calotowers");
  Electron_         = conf.getParameter<edm::InputTag> ("Electron");
  Photon_           = conf.getParameter<edm::InputTag> ("Photon");
  muon_             = conf.getParameter<edm::InputTag> ("muon");
  mctruth_          = conf.getParameter<edm::InputTag> ("mctruth");
  genEventScale_    = conf.getParameter<edm::InputTag> ("genEventScale");
  l1extramc_        = conf.getParameter<std::string>   ("l1extramc");
  l1extramu_        = conf.getParameter<std::string>   ("l1extramu");
  hltresults_       = conf.getParameter<edm::InputTag> ("hltresults");
  gtReadoutRecord_  = conf.getParameter<edm::InputTag> ("l1GtReadoutRecord");
  gtObjectMap_      = conf.getParameter<edm::InputTag> ("l1GtObjectMapRecord");
  gctCounts_        = conf.getParameter<edm::InputTag> ("l1GctCounts");
  MuCandTag2_       = conf.getParameter<edm::InputTag> ("MuCandTag2");
  MuIsolTag2_       = conf.getParameter<edm::InputTag> ("MuIsolTag2");
  MuCandTag3_       = conf.getParameter<edm::InputTag> ("MuCandTag3");
  MuIsolTag3_       = conf.getParameter<edm::InputTag> ("MuIsolTag3");
  MuLinkTag_        = conf.getParameter<edm::InputTag> ("MuLinkTag");
  HLTTau_           = conf.getParameter<edm::InputTag> ("HLTTau");

  m_file = 0;                  // set to null
  errCnt = 0;

  // read run parameters with a default value 
  edm::ParameterSet runParameters = conf.getParameter<edm::ParameterSet>("RunParameters");
  _HistName = runParameters.getUntrackedParameter<std::string>("HistogramFile", "test.root");
  _EtaMin   = runParameters.getUntrackedParameter<double>("EtaMin", -5.2);
  _EtaMax   = runParameters.getUntrackedParameter<double>("EtaMax",  5.2);

//   cout << "---------- Input Parameters ---------------------------" << endl;
//   cout << "  Output histograms written to: " << _HistName << std::endl;
//   cout << "  EtaMin: " << _EtaMin << endl;    
//   cout << "  EtaMax: " << _EtaMax << endl;    
//   cout << "  Monte:  " << _Monte << endl;    
//   cout << "  Debug:  " << _Debug << endl;    
//   cout << "-------------------------------------------------------" << endl;  

  // open the tree file
  m_file=new TFile(_HistName.c_str(), "RECREATE");
  m_file->cd();

  // Initialize the tree
  HltTree = new TTree("HltTree", "");

  // Setup the different analysis
  jet_analysis_.setup(conf, HltTree);
  bjet_analysis_.setup(conf, HltTree);
  elm_analysis_.setup(conf, HltTree);
  muon_analysis_.setup(conf, HltTree);
  mct_analysis_.setup(conf, HltTree);
  hlt_analysis_.setup(conf, HltTree);
  evt_header_.setup(HltTree);
}

// Boiler-plate "analyze" method declaration for an analyzer module.
void HLTAnalyzer::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {

  // To get information from the event setup, you must request the "Record"
  // which contains it and then extract the object you need
  //edm::ESHandle<CaloGeometry> geometry;
  //iSetup.get<IdealGeometryRecord>().get(geometry);

  // These declarations create handles to the types of records that you want
  // to retrieve from event "iEvent".
  edm::Handle<CaloJetCollection>                    recjets;
  edm::Handle<GenJetCollection>                     genjets;
  edm::Handle<CaloTowerCollection>                  caloTowers;
  edm::Handle<CaloMETCollection>                    recmet;
  edm::Handle<GenMETCollection>                     genmet;
  edm::Handle<METCollection>                        ht;
  edm::Handle<CandidateView>                        mctruth;
  edm::Handle<double>                               genEventScale;
  edm::Handle<GsfElectronCollection>                electrons;
  edm::Handle<PhotonCollection>                     photons;
  edm::Handle<MuonCollection>                       muon;
  edm::Handle<edm::TriggerResults>                  hltresults;
  edm::Handle<l1extra::L1EmParticleCollection>      l1extemi, l1extemn;
  edm::Handle<l1extra::L1MuonParticleCollection>    l1extmu;
  edm::Handle<l1extra::L1JetParticleCollection>     l1extjetc, l1extjetf, l1exttaujet;
  edm::Handle<l1extra::L1EtMissParticleCollection>  l1extmet;
  edm::Handle<L1GlobalTriggerReadoutRecord>         l1GtRR;
  edm::Handle<L1GlobalTriggerObjectMapRecord>       l1GtOMRec;
  edm::Handle<L1GlobalTriggerObjectMap>             l1GtOM;
  edm::Handle<L1GctJetCountsCollection>             l1GctCounts;
  edm::Handle<RecoChargedCandidateCollection>       mucands2, mucands3;
  edm::Handle<edm::ValueMap<bool> >                 isoMap2,  isoMap3;
  edm::Handle<MuonTrackLinksCollection>             mulinks;
  edm::Handle<reco::HLTTauCollection>               taus;

  // Extract objects (event fragments)
  // Reco Jets and Missing ET
  iEvent.getByLabel(recjets_, recjets);
  iEvent.getByLabel(genjets_, genjets);
  iEvent.getByLabel(recmet_,  recmet);
  iEvent.getByLabel(genmet_,  genmet);
  iEvent.getByLabel(calotowers_, caloTowers);
  iEvent.getByLabel(ht_, ht);
  // Reco Muons
  iEvent.getByLabel(muon_, muon);
  // Reco EGamma
  iEvent.getByLabel(Electron_, electrons);
  iEvent.getByLabel(Photon_, photons);
  // HLT results
  iEvent.getByLabel(hltresults_, hltresults);
  //  L1 Extra Info
  iEvent.getByLabel(l1extramc_, "Isolated", l1extemi);
  iEvent.getByLabel(l1extramc_, "NonIsolated", l1extemn);
  iEvent.getByLabel(l1extramu_, l1extmu);
  iEvent.getByLabel(l1extramc_, "Central", l1extjetc);
  iEvent.getByLabel(l1extramc_, "Forward", l1extjetf);
  iEvent.getByLabel(l1extramc_, "Tau", l1exttaujet);
  iEvent.getByLabel(l1extramc_, l1extmet);
  // L1 info
  iEvent.getByLabel(gtReadoutRecord_, l1GtRR);
  iEvent.getByLabel(gtObjectMap_, l1GtOMRec);
  iEvent.getByLabel(gctCounts_.label(), "", l1GctCounts);
  // MC info
  iEvent.getByLabel(genEventScale_, genEventScale );
  iEvent.getByLabel(mctruth_, mctruth);
  // OpenHLT info
  iEvent.getByLabel(MuCandTag2_, mucands2);
  iEvent.getByLabel(MuCandTag3_, mucands3);
  iEvent.getByLabel(MuIsolTag2_, isoMap2);
  iEvent.getByLabel(MuIsolTag3_, isoMap3);
  iEvent.getByLabel(MuLinkTag_,  mulinks);
  iEvent.getByLabel(HLTTau_, taus);

  
  // check the objects...
  typedef const char * MissingCollectionInfo;
  std::vector<MissingCollectionInfo> missing;

  if (not recjets.isValid()      ) { missing.push_back( kRecjets );       recjets.clear(); }
  if (not genjets.isValid()      ) { missing.push_back( kGenjets );       genjets.clear(); }
  if (not recmet.isValid()       ) { missing.push_back( kRecmet );        recmet.clear(); }
  if (not genmet.isValid()       ) { missing.push_back( kGenmet );        genmet.clear(); }
  if (not caloTowers.isValid()   ) { missing.push_back( kCaloTowers );    caloTowers.clear(); }
  if (not ht.isValid()           ) { missing.push_back( kHt );            ht.clear(); }
  if (not electrons.isValid()    ) { missing.push_back( kElectrons );     electrons.clear(); }
  if (not photons.isValid()      ) { missing.push_back( kPhotons );       photons.clear(); }
  if (not muon.isValid()         ) { missing.push_back( kMuon );          muon.clear(); }
  if (not taus.isValid()         ) { missing.push_back( kTaus );          taus.clear(); }

  if (not hltresults.isValid()   ) { missing.push_back( kHltresults );    hltresults.clear(); }
  if (not l1extemi.isValid()     ) { missing.push_back( kL1extemi );      l1extemi.clear(); }
  if (not l1extemn.isValid()     ) { missing.push_back( kL1extemn );      l1extemn.clear(); }
  if (not l1extmu.isValid()      ) { missing.push_back( kL1extmu );       l1extmu.clear(); }
  if (not l1extjetc.isValid()    ) { missing.push_back( kL1extjetc );     l1extjetc.clear(); }
  if (not l1extjetf.isValid()    ) { missing.push_back( kL1extjetf );     l1extjetf.clear(); }
  if (not l1exttaujet.isValid()  ) { missing.push_back( kL1exttaujet );   l1exttaujet.clear(); }
  if (not l1extmet.isValid()     ) { missing.push_back( kL1extmet );      l1extmet.clear(); }
  if (not l1GtRR.isValid()       ) { missing.push_back( kL1GtRR );        l1GtRR.clear(); }
  if (not l1GtOMRec.isValid()    ) { missing.push_back( kL1GtOMRec );     l1GtOMRec.clear(); }
  if (not l1GctCounts.isValid()  ) { missing.push_back( kL1GctCounts );   l1GctCounts.clear(); }
  
  if (not mctruth.isValid()      ) { missing.push_back( kMctruth );       mctruth.clear(); }
  if (not genEventScale.isValid()) { missing.push_back( kGenEventScale ); genEventScale.clear(); }

  if (not mucands2.isValid()     ) { missing.push_back( kMucands2 );      mucands2.clear(); }
  if (not mucands3.isValid()     ) { missing.push_back( kMucands3 );      mucands3.clear(); }
  if (not isoMap2.isValid()      ) { missing.push_back( kIsoMap2 );       isoMap2.clear(); }
  if (not isoMap3.isValid()      ) { missing.push_back( kIsoMap3 );       isoMap3.clear(); }
  if (not mulinks.isValid()      ) { missing.push_back( kMulinks );       mulinks.clear(); }

  if (not missing.empty() && (errCnt < errMax())) {
    errCnt++;
    std::stringstream out;       
    out <<  "OpenHLT analyer - missing collections:";
    BOOST_FOREACH(const MissingCollectionInfo entry, missing)
      out << "\n\t" << entry;
    edm::LogPrint("OpenHLT") << out.str() << std::endl; 
    if (errCnt == errMax())
      edm::LogWarning("OpenHLT") << "Maximum error count reached -- No more messages will be printed.";
  }

  // run the analysis, passing required event fragments
  jet_analysis_.analyze(recjets.product(), genjets.product(), recmet.product(), genmet.product(), ht.product(), taus.product(), caloTowers.product(), HltTree);
  muon_analysis_.analyze(muon.product(), mucands2.product(), isoMap2.product(), mucands3.product(), isoMap3.product(), mulinks.product(), HltTree);
  elm_analysis_.analyze(iEvent, iSetup, electrons.product(), photons.product(), HltTree);
  mct_analysis_.analyze(mctruth.product(), genEventScale.product(), HltTree);
  hlt_analysis_.analyze(hltresults.product(), l1extemi.product(), l1extemn.product(), l1extmu.product(), l1extjetc.product(), l1extjetf.product(), l1exttaujet.product(), l1extmet.product(), 
                        l1GtRR.product(), l1GtOMRec.product(), l1GctCounts.product(), HltTree);
  bjet_analysis_.analyze(iEvent, iSetup, HltTree);
  evt_header_.analyze(iEvent, HltTree);

  // std::cout << " Ending Event Analysis" << std::endl;
  // After analysis, fill the variables tree
  m_file->cd();
  HltTree->Fill();
}

// "endJob" is an inherited method that you may implement to do post-EOF processing and produce final output.
void HLTAnalyzer::endJob() {

  m_file->cd(); 
  HltTree->Write();
  delete HltTree;
  HltTree = 0;

  if (m_file!=0) { // if there was a tree file...
    m_file->Write(); // write out the branches
    delete m_file; // close and delete the file
    m_file=0; // set to zero to clean up
  }

}

