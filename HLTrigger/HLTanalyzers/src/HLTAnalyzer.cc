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

  Electron_   = conf.getParameter< std::string > ("Electron");
  Photon_     = conf.getParameter< std::string > ("Photon");
  muon_       = conf.getParameter< std::string > ("muon");

  mctruth_    = conf.getParameter< std::string > ("mctruth");
  genEventScale_ = conf.getParameter< std::string > ("genEventScale");
  l1extramc_  = conf.getParameter< std::string > ("l1extramc");
  hltresults_ = conf.getParameter< std::string > ("hltresults");
  //particleMapSource_ = conf.getParameter< std::string > ("particleMapSource");
  //particleMapSource_="";
  gtReadoutRecord_ = conf.getParameter< std::string > ("l1GtReadoutRecord");
  gtObjectMap_ = conf.getParameter< std::string > ("l1GtObjectMapRecord");

  //ecalDigisLabel_ = conf.getParameter<std::string> ("ecalDigisLabel");
  //hcalDigisLabel_ = conf.getParameter<std::string> ("hcalDigisLabel");

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
  //edm::ESHandle<CaloGeometry> geometry;
  //iSetup.get<IdealGeometryRecord>().get(geometry);

  // These declarations create handles to the types of records that you want
  // to retrieve from event "iEvent".
  edm::Handle<CaloJetCollection>  recjets,recjetsDummy;
  edm::Handle<GenJetCollection>  genjets,genjetsDummy;
  edm::Handle<CaloTowerCollection> caloTowers,caloTowersDummy;
  edm::Handle<CaloMETCollection> recmet, recmetDummy;
  edm::Handle<GenMETCollection> genmet,genmetDummy;
  edm::Handle<METCollection> ht,htDummy;
  // edm::Handle<edm::HepMCProduct> hepmcHandle;
  edm::Handle<CandidateView> mctruth,mctruthDummy;
  //edm::Handle<CandidateCollection> mctruth,mctruthDummy;
  edm::Handle< double > genEventScale;
  edm::Handle<PixelMatchGsfElectronCollection> Electron, ElectronDummy;
  edm::Handle<PhotonCollection> Photon, PhotonDummy;
  edm::Handle<MuonCollection> muon,muonDummy;
//   edm::Handle<HLTFilterObjectWithRefs> hltobj;
  edm::Handle<edm::TriggerResults> hltresults,hltresultsDummy;
  edm::Handle<l1extra::L1EmParticleCollection> l1extemi,l1extemn,l1extemiDummy,l1extemnDummy;
  edm::Handle<l1extra::L1MuonParticleCollection> l1extmu, l1extmuDummy;
  edm::Handle<l1extra::L1JetParticleCollection> l1extjetc,l1extjetf,l1exttaujet,l1extjetcDummy,l1extjetfDummy,l1exttaujetDummy;
  edm::Handle<l1extra::L1EtMissParticleCollection> l1extmet, l1extmetDummy;
//edm::Handle<l1extra::L1ParticleMapCollection> l1mapcoll;
  edm::Handle<L1GlobalTriggerReadoutRecord> l1GtRR;
  edm::Handle<L1GlobalTriggerObjectMapRecord> l1GtOMRec;
  edm::Handle<L1GlobalTriggerObjectMap> l1GtOM;
//  edm::Handle<EcalTrigPrimDigiCollection> ecal;
//  edm::Handle<HcalTrigPrimDigiCollection> hcal;

  
  // ccla double pthat = *genEventScale;


  // Extract Data objects (event fragments)
  //Jets and Missing ET
  iEvent.getByLabel(recjets_,recjets);
  iEvent.getByLabel(genjets_,genjets);
  iEvent.getByLabel(recmet_,recmet);
  iEvent.getByLabel (genmet_,genmet);
  iEvent.getByLabel(calotowers_,caloTowers);
  iEvent.getByLabel(ht_,ht);
  // Egamma
  iEvent.getByLabel(Electron_,Electron);
  iEvent.getByLabel(Photon_,Photon);
  // Muons
  iEvent.getByLabel(muon_,muon);
  // HLT 
  iEvent.getByLabel(hltresults_,hltresults);
  //  L1 Extra Info
  iEvent.getByLabel(l1extramc_,"Isolated",l1extemi);
  iEvent.getByLabel(l1extramc_,"NonIsolated",l1extemn);
  iEvent.getByLabel(l1extramc_,l1extmu);
  iEvent.getByLabel(l1extramc_,"Central",l1extjetc);
  iEvent.getByLabel(l1extramc_,"Forward",l1extjetf);
  iEvent.getByLabel(l1extramc_,"Tau",l1exttaujet);
  iEvent.getByLabel(l1extramc_,l1extmet);
  // MC info
  iEvent.getByLabel(genEventScale_, genEventScale );
  iEvent.getByLabel(mctruth_,mctruth);
  // iEvent.getByLabel("VtxSmeared", "", hepmcHandle);  no longer used

  // check the objects...
  string errMsg("");
  if (! recjets.isValid()    ) { errMsg=errMsg + "  -- No RecJets"; recjets = recjetsDummy;}
  if (! genjets.isValid()    ) { errMsg=errMsg + "  -- No GenJets"; genjets=genjetsDummy;}
  if (! recmet.isValid()     ) { errMsg=errMsg + "  -- No RecMET"; recmet = recmetDummy;}
  if (! genmet.isValid()     ) { errMsg=errMsg + "  -- No GenMet"; genmet=genmetDummy;}
  if (! caloTowers.isValid() ) { errMsg=errMsg + "  -- No CaloTowers"; caloTowers=caloTowersDummy;}
  if (! ht.isValid()         ) { errMsg=errMsg + "  -- No HT"; ht = htDummy;}
  if (! Electron.isValid()   ) { errMsg=errMsg + "  -- No Candidate Electrons"; Electron=ElectronDummy;}

  if (! Photon.isValid()     ) { errMsg=errMsg + "  -- No Candidate Photons"; Photon=PhotonDummy;}
  if (! muon.isValid()       ) { errMsg=errMsg + "  -- No Candidate Muons"; muon=muonDummy;}

  if (! hltresults.isValid() ) { errMsg=errMsg + "  -- No HLTRESULTS"; hltresults=hltresultsDummy;}

  if (! l1extemi.isValid()   ) { errMsg=errMsg + "  -- No Isol. L1Em objects"; l1extemi = l1extemiDummy;}
  if (! l1extemn.isValid()   ) { errMsg=errMsg + "  -- No Non-isol. L1Em objects"; l1extemn = l1extemnDummy;}
  if (! l1extmu.isValid()    ) { errMsg=errMsg + "  -- No L1Mu objects"; l1extmu = l1extmuDummy;}
  if (! l1extjetc.isValid()  ) { errMsg=errMsg + "  -- No central L1Jet objects"; l1extjetc = l1extjetcDummy;}
  if (! l1extjetf.isValid()  ) { errMsg=errMsg + "  -- No forward L1Jet objects"; l1extjetf = l1extjetfDummy;}
  if (! l1exttaujet.isValid()) { errMsg=errMsg + "  -- No L1Jet-Tau objects"; l1exttaujet = l1exttaujetDummy;}
  if (! l1extmet.isValid()   ) { errMsg=errMsg + "  -- No L1EtMiss object"; l1extmet = l1extmetDummy;}
//try {iEvent.getByLabel(particleMapSource_,l1mapcoll );} catch (...) { errMsg=errMsg + "  -- No L1 Map Collection";}
  try {iEvent.getByLabel(gtReadoutRecord_,l1GtRR);} catch (...) { errMsg=errMsg + "  -- No L1 GT ReadouRecord";}
  try {iEvent.getByLabel(gtObjectMap_,l1GtOMRec);} catch (...) { errMsg=errMsg + "  -- No L1 GT ObjectMap";}

  if (! mctruth.isValid()    ) { errMsg=errMsg + "  -- No Gen Particles"; mctruth = mctruthDummy;}


  //HepMC::GenEvent hepmc,hepmcDummy;
  //if (hepmcHandle.isValid()){
  //  hepmc = hepmcHandle->getHepMCData(); 
  //} else                       { errMsg=errMsg + "  -- No MCTruth"; hepmc = hepmcDummy;}


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
  jet_analysis_.analyze(*recjets,*genjets, *recmet,*genmet, *ht, *caloTowers, HltTree);
  elm_analysis_.analyze(*Electron, *Photon, HltTree);
  muon_analysis_.analyze(*muon, HltTree);
  mct_analysis_.analyze(*mctruth,*genEventScale,HltTree);
  // hlt_analysis_.analyze(/**hltobj,*/*hltresults,*l1extemi,*l1extemn,*l1extmu,*l1extjetc,*l1extjetf,*l1exttaujet,*l1extmet,/* *l1mapcoll, */HltTree);
  hlt_analysis_.analyze(*hltresults,*l1extemi,*l1extemn,*l1extmu,*l1extjetc,*l1extjetf,*l1exttaujet,*l1extmet,
			*l1GtRR.product(),*l1GtOMRec.product(),HltTree);
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

