// File: JetAnalyzer.cc
// Description:  Example of Jet Analysis driver originally from Jeremy Mans, 
//               developed by Lenny Apanesevich and Anwar Bhatti for various purposes.
// Date:  31-August-2006

#include "RecoJets/JetAnalyzers/interface/JetAnalyzer.h"

// Boiler-plate constructor definition of an analyzer module:
//
JetAnalyzer::JetAnalyzer(edm::ParameterSet const& conf) {

  // If your module takes parameters, here is where you would define
  // their names and types, and access them to initialize internal
  // variables. Example as follows:
  //
  std::cout << " Beginning JetAnalyzer Analysis " << std::endl;

  recjets_    = conf.getParameter< std::string > ("recjets");
  genjets_    = conf.getParameter< std::string > ("genjets");
  recmet_     = conf.getParameter< std::string > ("recmet");
  genmet_     = conf.getParameter< std::string > ("genmet");
  calotowers_ = conf.getParameter< std::string > ("calotowers");

  errCnt=0;

  analysis_.setup(conf);

}

// Boiler-plate "analyze" method declaration for an analyzer module.
//
void JetAnalyzer::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {


  // To get information from the event setup, you must request the "Record"
  // which contains it and then extract the object you need
  edm::ESHandle<CaloGeometry> geometry;
  iSetup.get<IdealGeometryRecord>().get(geometry);

  // These declarations create handles to the types of records that you want
  // to retrieve from event "iEvent".
  //

  edm::Handle<CaloJetCollection>  recjets;
  edm::Handle<GenJetCollection>  genjets;
  edm::Handle<CaloTowerCollection> caloTowers;
  edm::Handle<CaloMETCollection> recmet;
  edm::Handle<METCollection> genmet;
  edm::Handle<edm::HepMCProduct> mctruthHandle;

  edm::Handle<HBHERecHitCollection> hbhe_hits;
  edm::Handle<HBHEDigiCollection> hbhe_digis;
  edm::Handle<HORecHitCollection> ho_hits;
  edm::Handle<HODigiCollection> ho_digis;
  edm::Handle<HFRecHitCollection> hf_hits;
  edm::Handle<HFDigiCollection> hf_digis;
  edm::Handle<HcalTBTriggerData> trigger;

  // Data objects
  iEvent.getByLabel (recjets_,recjets);
  iEvent.getByLabel (recmet_,recmet);
  iEvent.getByLabel (recmet_,recmet);
  iEvent.getByLabel (calotowers_,caloTowers);

  string errMsg("");
  try {
    iEvent.getByType(hbhe_hits);
  } catch (...) {
    errMsg=errMsg + "  -- No HBHE hits";
  }

  try {
    iEvent.getByType(hbhe_digis);
  } catch (...) {
    errMsg=errMsg + "  -- No HBHE digis";
  }

  try {
    iEvent.getByType(ho_hits);
  } catch (...) {
    errMsg=errMsg + "  -- No HO hits";
  }

  try {
    iEvent.getByType(ho_digis);
  } catch (...) {
    errMsg=errMsg + "  -- No HO digis";
  }

  try {
    iEvent.getByType(hf_hits);
  } catch (...) {
    errMsg=errMsg + "  -- No HF hits";
  }

  try {
    iEvent.getByType(hf_digis);
  } catch (...) {
    errMsg=errMsg + "  -- No HF digis";
  }

  // Trigger Information
  try {
    iEvent.getByType(trigger);
  } catch (...) {
    errMsg=errMsg + "  -- No TB Trigger info";
  }


  // MC objects
  HepMC::GenEvent mctruth;
  try {
    iEvent.getByLabel("VtxSmeared", "", mctruthHandle);
    mctruth = mctruthHandle->getHepMCData();
  
  } catch (...) {
    errMsg=errMsg + "  -- No MC truth";
  }

  try {
    iEvent.getByLabel (genjets_,genjets);
  } catch (...) {
    errMsg=errMsg + "  -- No GenJets";
  }

  try {
    iEvent.getByLabel (genmet_,genmet);
  } catch (...) {
    errMsg=errMsg + "  -- GenMet";
  }

  if ((errMsg != "") && (errCnt < errMax())){
    errCnt=errCnt+1;
    errMsg=errMsg + ".";
    std::cout << "%JetAnalyzer-Warning" << errMsg << std::endl;
    if (errCnt == errMax()){
      errMsg="%JetAnalyzer-Warning -- Maximum error count reached -- No more messages will be printed.";
      std::cout << errMsg << std::endl;    
    }
  }
  // "do stuff"
  //
  analysis_.analyze(*recjets,*genjets,
		    *recmet,*genmet,*caloTowers,mctruth,
		    *hbhe_hits,*hbhe_digis,*ho_hits,*ho_digis,*hf_hits,*hf_digis,
		    *trigger,*geometry);
  //analysis_.dummyAnalyze(*geometry);
}

// "endJob" is an inherited method that you may implement to do post-EOF processing
// and produce final output.
//
void JetAnalyzer::endJob() {
  analysis_.done();
}

