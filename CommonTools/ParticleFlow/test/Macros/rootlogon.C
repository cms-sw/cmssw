
void rootlogon() {
  gROOT->SetStyle("Plain");
  gStyle->SetPalette(1);
  gSystem->Load("libPhysics");
  gSystem->Load("libEG");
  gStyle->SetHistMinimumZero(kTRUE);
}

void loadFWLite() {
  gSystem->Load("libFWCoreFWLite.so");
  AutoLibraryLoader::enable();
}

TTree* getEventsrootlogon() {
  TTree* events = 0;
  gDirectory->GetObject("Events", events);
  return events;
}

void initAOD(const char* process) {

  string verticesAod = "recoVertexs_offlinePrimaryVertices__"; 
  verticesAod += process;

  string pfCandidatesAod = "recoPFCandidates_particleFlow__"; 
  pfCandidatesAod += process;

  string ic5GenJetsAod = "recoGenJets_iterativeCone5GenJets__";  
  ic5GenJetsAod += process;

  string pfJetsAod = "recoPFJets_iterativeCone5PFJets__";  
  pfJetsAod += process;

  TTree* Events = getEventsrootlogon();
  Events->SetAlias("verticesAod", verticesAod.c_str()); 
  Events->SetAlias("pfCandidatesAod",  pfCandidatesAod.c_str());
  Events->SetAlias("ic5GenJetsAod",  ic5GenJetsAod.c_str());
  Events->SetAlias("pfJetsAod",  pfJetsAod.c_str());

}

void initPF2PAT(const char* process) {


  string met = "recoMETs_pfMET__"; met += process;
  string pu = "recoPileUpPFCandidates_pfPileUp__";  pu+= process;
  string jetsin = "recoPFJets_pfJets__"; jetsin += process;
  string jetsout = "recoPFJets_pfNoTau__"; jetsout += process;
  string taus = "recoPFTaus_allLayer0Taus__"; taus += process;
  string muons = "recoPFCandidates_pfIsolatedMuons__"; muons += process;
  string electrons = "recoPFCandidates_pfIsolatedElectrons__"; electrons += process;
  string pfcandout = "recoPFCandidates_pfNoJet__"; pfcandout += process;  
  string noPileUp = "recoPFCandidates_pfNoPileUp__"; noPileUp += process;  


  string genMetTrue = "recoGenMETs_genMetTrue__";
  genMetTrue += process;
  string decaysFromZs = "recoGenParticles_decaysFromZs__";
  decaysFromZs += process;

  TTree* Events = getEventsrootlogon();
  Events->SetAlias("met", met.c_str() );
  Events->SetAlias("pileUp", pu.c_str() );
  Events->SetAlias("jetsAll", jetsin.c_str() );
  Events->SetAlias("jets", jetsout.c_str() );
  Events->SetAlias("taus", taus.c_str());
  Events->SetAlias("muons", muons.c_str());
  Events->SetAlias("electrons", electrons.c_str());
  Events->SetAlias("pfCandOut", pfcandout.c_str());
  Events->SetAlias("noPileUp", noPileUp.c_str());

  Events->SetAlias("genmet",  genMetTrue.c_str());
  Events->SetAlias("decaysFromZs",  decaysFromZs.c_str());
}


void initPAT(const char* process) {
  string taus = "patTaus_selectedPatTausPFlow__"; taus += process;
  string jets = "patJets_selectedPatJetsPFlow__"; jets += process;
  string met = "patMETs_patMETsPFlow__";  met+= process;
  string mus = "patMuons_selectedPatMuonsPFlow__"; mus += process;
  string eles = "patElectrons_selectedPatElectronsPFlow__"; eles += process;
  
  string patCaloJets = "patJets_selectedPatJets__"; patCaloJets += process;

  TTree* Events = getEventsrootlogon();
  Events->SetAlias("patTaus", taus.c_str() );
  Events->SetAlias("patJets", jets.c_str() );
  Events->SetAlias("patCaloJets", patCaloJets.c_str() );
  Events->SetAlias("patMet", met.c_str() );
  Events->SetAlias("patMuons", mus.c_str() );
  Events->SetAlias("patElectrons", eles.c_str() );
}

