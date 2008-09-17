// File: HLTAnalyzer.cc
// Description:  Example of Analysis driver originally from Jeremy Mans, 
// Date:  13-October-2006

#include "HLTrigger/HLTanalyzers/interface/HLTAnalyzer.h"

static const char * kRecjets                     = "RecJets";
static const char * kGenjets                     = "GenJets";
static const char * kRecmet                      = "RecMET";
static const char * kGenmet                      = "GenMet";
static const char * kCaloTowers                  = "CaloTowers";
static const char * kHt                          = "HT";
static const char * kElectrons                   = "Candidate Electrons";
static const char * kPhotons                     = "Candidate Photons";
static const char * kMuon                        = "Candidate Muons";
static const char * kHltresults                  = "HLTRESULTS";
static const char * kL1extemi                    = "Isol. L1Em objects";
static const char * kL1extemn                    = "Non-isol. L1Em objects";
static const char * kL1extmu                     = "L1Mu objects";
static const char * kL1extjetc                   = "central L1Jet objects";
static const char * kL1extjetf                   = "forward L1Jet objects";
static const char * kL1exttaujet                 = "L1Jet-Tau objects";
static const char * kL1extmet                    = "L1EtMiss object";
static const char * kL1GtRR                      = "L1 GT ReadouRecord";
static const char * kL1GtOMRec                   = "L1 GT ObjectMap";
static const char * kL1GctCounts                 = "L1 GCT JetCount Digis";
static const char * kMctruth                     = "Gen Particles";
static const char * kGenEventScale               = "Event Scale";
static const char * kMucands2                    = "L2 muon candidates";
static const char * kMucands3                    = "L3 muon candidates";
static const char * kIsoMap2                     = "L2 muon isolation map";
static const char * kIsoMap3                     = "L3 muon isolation map";
static const char * kMulinks                     = "L3 muon link";
static const char * kTaus                        = "Tau candidates";
static const char * kBTagJets                    = "L2 b-jet collection";
static const char * kBTagCorrectedJets           = "L2 calibrated b-jet collection";
static const char * kBTagLifetimeBJetsL25        = "L2.5 b-jet lifetime tagsL";
static const char * kBTagLifetimeBJetsL3         = "L3 b-jet lifetime tagsL";
static const char * kBTagLifetimeBJetsL25Relaxed = "L2.5 b-jet lifetime tagsL (relaxed)";
static const char * kBTagLifetimeBJetsL3Relaxed  = "L3 b-jet lifetime tagsL (relaxed)";
static const char * kBTagSoftmuonBJetsL25        = "L2.5 b-jet soft muon tagsL";
static const char * kBTagSoftmuonBJetsL3         = "L3 b-jet soft muon tagsL";
static const char * kBTagPerformanceBJetsL25     = "L2.5 b-jet perf. meas. tag";
static const char * kBTagPerformanceBJetsL3      = "L3 b-jet perf. meas. tag";


// Boiler-plate constructor definition of an analyzer module:
HLTAnalyzer::HLTAnalyzer(edm::ParameterSet const& conf) {

  //set parameter defaults 
  _EtaMin = -5.2;
  _EtaMax =  5.2;
  m_file  = 0;                  // set to null
  _HistName = "test.root";

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

  errCnt = 0;

  edm::ParameterSet myAnaParams = conf.getParameter<edm::ParameterSet> ("RunParameters") ;
  vector<std::string> parameterNames = myAnaParams.getParameterNames() ;
  for ( vector<std::string>::iterator iParam = parameterNames.begin();
        iParam != parameterNames.end(); iParam++ ){
    if ( (*iParam) == "HistogramFile" ) _HistName =  myAnaParams.getParameter<string>( *iParam );
    else if ( (*iParam) == "EtaMin" ) _EtaMin =  myAnaParams.getParameter<double>( *iParam );
    else if ( (*iParam) == "EtaMax" ) _EtaMax =  myAnaParams.getParameter<double>( *iParam );
  }

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
  string errMsg("");
  if (! recjets.isValid()    ) { errMsg += "  -- No RecJets";                recjets.clear(); }
  if (! genjets.isValid()    ) { errMsg += "  -- No GenJets";                genjets.clear(); }
  if (! recmet.isValid()     ) { errMsg += "  -- No RecMET";                 recmet.clear(); }
  if (! genmet.isValid()     ) { errMsg += "  -- No GenMet";                 genmet.clear(); }
  if (! caloTowers.isValid() ) { errMsg += "  -- No CaloTowers";             caloTowers.clear(); }
  if (! ht.isValid()         ) { errMsg += "  -- No HT";                     ht.clear(); }
  if (! electrons.isValid()  ) { errMsg += "  -- No Candidate Electrons";    electrons.clear(); }
  if (! photons.isValid()    ) { errMsg += "  -- No Candidate Photons";      photons.clear(); }
  if (! muon.isValid()       ) { errMsg += "  -- No Candidate Muons";        muon.clear(); }

  if (! hltresults.isValid() ) { errMsg += "  -- No HLTRESULTS";             hltresults.clear(); }
  if (! l1extemi.isValid()   ) { errMsg += "  -- No Isol. L1Em objects";     l1extemi.clear(); }
  if (! l1extemn.isValid()   ) { errMsg += "  -- No Non-isol. L1Em objects"; l1extemn.clear(); }
  if (! l1extmu.isValid()    ) { errMsg += "  -- No L1Mu objects";           l1extmu.clear(); }
  if (! l1extjetc.isValid()  ) { errMsg += "  -- No central L1Jet objects";  l1extjetc.clear(); }
  if (! l1extjetf.isValid()  ) { errMsg += "  -- No forward L1Jet objects";  l1extjetf.clear(); }
  if (! l1exttaujet.isValid()) { errMsg += "  -- No L1Jet-Tau objects";      l1exttaujet.clear(); }
  if (! l1extmet.isValid()   ) { errMsg += "  -- No L1EtMiss object";        l1extmet.clear(); }
  if (! l1GtRR.isValid()     ) { errMsg += "  -- No L1 GT ReadouRecord";     l1GtRR.clear(); }
  if (! l1GtOMRec.isValid()  ) { errMsg += "  -- No L1 GT ObjectMap";        l1GtOMRec.clear(); }
  if (! l1GctCounts.isValid()) { errMsg += "  -- No L1 GCT JetCount Digis";  l1GctCounts.clear(); }
  
  if (! mctruth.isValid()    ) { errMsg += "  -- No Gen Particles";          mctruth.clear(); }
  if (! genEventScale.isValid()) { errMsg += "  -- No Event Scale";          genEventScale.clear(); }

  if (! mucands2.isValid()   ) { errMsg += "  -- No L2 muon candidates";     mucands2.clear(); }
  if (! mucands3.isValid()   ) { errMsg += "  -- No L3 muon candidates";     mucands3.clear(); }
  if (! isoMap2.isValid()    ) { errMsg += "  -- No L2 muon isolation map";  isoMap2.clear(); }
  if (! isoMap3.isValid()    ) { errMsg += "  -- No L3 muon isolation map";  isoMap3.clear(); }
  if (! mulinks.isValid()    ) { errMsg += "  -- No L3 muon link";           mulinks.clear(); }

  if (! taus.isValid()       ) { errMsg += "  -- No Tau candidates";         taus.clear(); }

  if ((errMsg != "") && (errCnt < errMax())){
    errCnt=errCnt+1;
    errMsg=errMsg + ".";
    std::cout << "%HLTAnalyzer-Warning" << errMsg << std::endl;
    if (errCnt == errMax()){
      errMsg="%HLTAnalyzer-Warning -- Maximum error count reached -- No more messages will be printed.";
      std::cout << errMsg << std::endl;    
    }
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

