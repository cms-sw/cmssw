// File: HLTAnalyzer.cc
// Description:  Example of Analysis driver originally from Jeremy Mans,
// Date:  13-October-2006

#include "HLTrigger/HLTanalyzers/interface/HLTAnalyzer.h"

// Boiler-plate constructor definition of an analyzer module:
HLTAnalyzer::HLTAnalyzer(edm::ParameterSet const& conf) {

  //set parameter defaults 
  _EtaMin=-5.2;
  _EtaMax=5.2;
  _HistName="test.root";
  m_file=0; // set to null

  // If your module takes parameters, here is where you would define
  // their names and types, and access them to initialize internal
  // variables. Example as follows:
  std::cout << " Beginning HLTAnalyzer Analysis " << std::endl;

  recjets_    = conf.getParameter< std::string > ("recjets");
  genjets_    = conf.getParameter< std::string > ("genjets");
  recmet_     = conf.getParameter< std::string > ("recmet");
  genmet_     = conf.getParameter< std::string > ("genmet");
  ht_         = conf.getParameter< std::string > ("ht");
  calotowers_ = conf.getParameter< std::string > ("calotowers");

  Electron_    = conf.getParameter< std::string > ("Electron");
  Photon_    = conf.getParameter< std::string > ("Photon");
  muon_    = conf.getParameter< std::string > ("muon");

  mctruth_   = conf.getParameter< std::string > ("mctruth");
//   hltobj_    = conf.getParameter< std::string > ("hltobj");

  l1extramc_ = conf.getParameter< std::string > ("l1extramc");

  particleMapSource_ = conf.getParameter< std::string > ("particleMapSource");

  ecalDigisLabel_ = conf.getParameter<std::string> ("ecalDigisLabel");
  hcalDigisLabel_ = conf.getParameter<std::string> ("hcalDigisLabel");

  errCnt=0;

  edm::ParameterSet myAnaParams = conf.getParameter<edm::ParameterSet>("RunParameters") ;
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
  m_file=new TFile(_HistName.c_str(),"RECREATE");
  m_file->cd();

  // Initialize the tree
  HltTree = 0;
  HltTree = new TTree("HltTree","");

  // Setup the different analysis
  jet_analysis_.setup(conf, HltTree);
  elm_analysis_.setup(conf, HltTree);
  muon_analysis_.setup(conf, HltTree);
  mct_analysis_.setup(conf, HltTree);
  hlt_analysis_.setup(conf, HltTree);

}

// Boiler-plate "analyze" method declaration for an analyzer module.
void HLTAnalyzer::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {

  // To get information from the event setup, you must request the "Record"
  // which contains it and then extract the object you need
  edm::ESHandle<CaloGeometry> geometry;
  iSetup.get<IdealGeometryRecord>().get(geometry);

  // These declarations create handles to the types of records that you want
  // to retrieve from event "iEvent".
  edm::Handle<CaloJetCollection>  recjets;
  edm::Handle<GenJetCollection>  genjets;
  edm::Handle<CaloTowerCollection> caloTowers;
  edm::Handle<CaloMETCollection> recmet;
  edm::Handle<GenMETCollection> genmet;
  edm::Handle<METCollection> ht;
  edm::Handle<edm::HepMCProduct> hepmcHandle;
  edm::Handle<CandidateCollection> mctruth;
  edm::Handle<ElectronCollection> Electron;
  edm::Handle<PhotonCollection> Photon;
  edm::Handle<MuonCollection> muon;
//   edm::Handle<HLTFilterObjectWithRefs> hltobj;
  edm::Handle<edm::TriggerResults> hltresults;
  edm::Handle<l1extra::L1EmParticleCollection> l1extemi,l1extemn;
  edm::Handle<l1extra::L1MuonParticleCollection> l1extmu;
  edm::Handle<l1extra::L1JetParticleCollection> l1extjetc,l1extjetf,l1exttaujet;
  edm::Handle<l1extra::L1EtMissParticle> l1extmet;
  edm::Handle<l1extra::L1ParticleMapCollection> l1mapcoll;
  edm::Handle<EcalTrigPrimDigiCollection> ecal;
  edm::Handle<HcalTrigPrimDigiCollection> hcal;


  // Extract Data objects (event fragments)
  // make sure to catch exceptions if they don't exist...
  string errMsg("");
  try {iEvent.getByLabel(recjets_,recjets);} catch (...) { errMsg=errMsg + "  -- No RecJets";}
  try {iEvent.getByLabel(recmet_,recmet);} catch (...) {errMsg=errMsg + "  -- No RecMET";}
  try {iEvent.getByLabel(calotowers_,caloTowers);} catch (...) {errMsg=errMsg + "  -- No CaloTowers";}
  try {iEvent.getByLabel (genjets_,genjets);} catch (...) { errMsg=errMsg + "  -- No GenJets";}
  try {iEvent.getByLabel (ht_,ht);} catch (...) { errMsg=errMsg + "  -- No HT";}
  try {iEvent.getByLabel (genmet_,genmet);} catch (...) { errMsg=errMsg + "  -- No GenMet";}
  try {iEvent.getByLabel(Electron_,Electron);} catch (...) { errMsg=errMsg + "  -- No Candidate Electrons";}
  try {iEvent.getByLabel(Photon_,Photon);} catch (...) { errMsg=errMsg + "  -- No Candidate Photons";}
  try {iEvent.getByLabel(muon_,muon);} catch (...) { errMsg=errMsg + "  -- No Candidate Muons";}
//   try {iEvent.getByLabel(hltobj_,hltobj);} catch (...) { errMsg=errMsg + "  -- No HLTOBJ";}
  try {iEvent.getByType(hltresults);} catch (...) { errMsg=errMsg + "  -- No HLTRESULTS";}
  try {iEvent.getByLabel(l1extramc_,"Isolated",l1extemi);} catch (...) { errMsg=errMsg + "  -- No Isol. L1Em objects";}
  try {iEvent.getByLabel(l1extramc_,"NonIsolated",l1extemn);} catch (...) { errMsg=errMsg + "  -- No Non-isol. L1Em objects";}
  try {iEvent.getByLabel(l1extramc_,l1extmu);} catch (...) { errMsg=errMsg + "  -- No L1Mu objects";}
  try {iEvent.getByLabel(l1extramc_,"Central",l1extjetc);} catch (...) { errMsg=errMsg + "  -- No central L1Jet objects";}
  try {iEvent.getByLabel(l1extramc_,"Forward",l1extjetf);} catch (...) { errMsg=errMsg + "  -- No forward L1Jet objects";}
  try {iEvent.getByLabel(l1extramc_,"Tau",l1exttaujet);} catch (...) { errMsg=errMsg + "  -- No L1Jet-Tau objects";}
  try {iEvent.getByLabel(l1extramc_,l1extmet);} catch (...) { errMsg=errMsg + "  -- No L1EtMiss object";}
  try {iEvent.getByLabel(particleMapSource_,l1mapcoll );} catch (...) { errMsg=errMsg + "  -- No L1 Map Collection";}

  try {iEvent.getByLabel(mctruth_,mctruth);} catch (...) { errMsg=errMsg + "  -- No Gen Particles";}

  try {iEvent.getByLabel(ecalDigisLabel_,ecal);} catch (...) { errMsg=errMsg + "  -- No ECAL TriggPrim";}
  try {iEvent.getByLabel(hcalDigisLabel_,hcal);} catch (...) { errMsg=errMsg + "  -- No HCAL TriggPrim";}

  HepMC::GenEvent hepmc;
  try {
//     iEvent.getByLabel("VtxSmeared", "", hepmcHandle);
    iEvent.getByLabel("source", "", hepmcHandle);
    hepmc = hepmcHandle->getHepMCData(); 
  } catch (...) { errMsg=errMsg + "  -- No MC truth"; }

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
  jet_analysis_.analyze(*recjets,*genjets, *recmet,*genmet, *ht, *caloTowers, *geometry, HltTree);
  elm_analysis_.analyze(*Electron, *Photon, *geometry, HltTree);
  muon_analysis_.analyze(*muon, *geometry, HltTree);
  mct_analysis_.analyze(*mctruth,hepmc,HltTree);
  hlt_analysis_.analyze(/**hltobj,*/*hltresults,*l1extemi,*l1extemn,*l1extmu,*l1extjetc,*l1extjetf,*l1exttaujet,*l1extmet,*l1mapcoll,HltTree);

  // After analysis, fill the variables tree
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

