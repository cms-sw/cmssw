#include "DQM/Physics/src/CentralityDQM.h"

#include <memory>

// DQM
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Centrality
#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
#include "DataFormats/HeavyIonEvent/interface/CentralityProvider.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

using namespace edm;
using namespace reco; 


//
// -- Constructor
//
CentralityDQM::CentralityDQM(const edm::ParameterSet& ps){

  edm::LogInfo("CentralityDQM") <<  " Starting CentralityDQM " << "\n" ;

  bei_ = Service<DQMStore>().operator->();
  bei_->setCurrentFolder("Physics/Centrality");
  bookHistos(bei_);

  centrality_ = ps.getParameter<edm::InputTag>("centrality_");


  // just to initialize
}


//
// -- Destructor
//
CentralityDQM::~CentralityDQM(){
  edm::LogInfo("CentralityDQM") <<  " Deleting CentralityDQM " << "\n" ;
}


//
// -- Begin Job
//
void CentralityDQM::beginJob(){
  nLumiSecs_ = 0;
}


//
// -- Begin Run
//
void CentralityDQM::beginRun(Run const& run, edm::EventSetup const& eSetup) {
  edm::LogInfo ("CentralityDQM") <<"[CentralityDQM]: Begining of Run";
}


// -- Begin  Luminosity Block
//
void CentralityDQM::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) { 
  //edm::LogInfo ("CentralityDQM") <<"[CentralityDQM]: Begin of LS transition";
}

         
//
//  -- Book histograms
//
void CentralityDQM::bookHistos(DQMStore* bei){
 
  bei->cd();
  
  bei->setCurrentFolder("Physics/Centrality");
  
  h_hiNpix= bei->book1D("h_hiNpix", "h_hiNpix", 1000,0,1000 );
  h_hiNpixelTracks= bei->book1D("h_hiNpixelTracks", "hiNpixelTracks",1000,0,1000 );
  h_hiNtracks= bei->book1D("h_hiNtracks", "h_hiNtracks", 1000,0,1000); 

  h_hiNtracksPtCut= bei->book1D("h_hiNtracksPtCut", "h_hiNtracksPtCut", 1000,0,1000 );
  h_hiNtracksEtaCut= bei->book1D("h_hiNtracksEtaCut", "h_hiNtracksEtaCut",1000,0,1000 );
  h_hiNtracksEtaPtCut= bei->book1D("h_hiNtracksEtaPtCut", "h_hiNtracksEtaPtCut",1000,0,1000 );
  h_hiHF= bei->book1D("h_hiHF", "h_hiHF", 1000,0,1000 );
  h_hiHFplus= bei->book1D("h_hiHFplus", "h_hiHFplus", 1000,0,1000 ); 
  h_hiHFminus= bei->book1D("h_hiHFminus", "h_hiHFminus", 1000,0,1000); 
  h_hiHFplusEta4= bei->book1D("h_hiHFplusEta4", "h_hiHFplusEta4", 1000,0,1000);
  h_hiHFminusEta4= bei->book1D("h_hiHFminusEta4", "h_hiHFminusEta4",1000,0,1000 ); 
  h_hiHFhit= bei->book1D("h_hiHFhit", "h_hiHFhit",1000,0,1000 );
  h_hiHFhitPlus= bei->book1D("h_hiHFhitPlus", "h_hiHFhitPlus",1000,0,1000 ); 
  h_hiHFhitMinus= bei->book1D("h_hiHFhitMinus", "h_hiHFhitMinus",1000,0,1000 );
  h_hiEB= bei->book1D("h_hiEB", "h_hiEB",1000,0,1000 );
  h_hiET= bei->book1D("h_hiET", "h_hiET",1000,0,1000 );
  h_hiEE= bei->book1D("h_hiEE", "h_hiEE",1000,0,1000); 
  h_hiEEplus= bei->book1D("h_hiEEplus", "h_hiEEplus", 1000,0,1000); 
  h_hiEEminus= bei->book1D("h_hiEEminus", "h_hiEEminus",1000,0,1000 );
  h_hiZDC= bei->book1D("h_hiZDC", "h_hiZDC",1000,0,1000 ); 
  h_hiZDCplus= bei->book1D("h_hiZDCplus", "h_hiZDCplus",1000,0,1000 ); 
  h_hiZDCminus= bei->book1D("h_hiZDCminus", "h_hiZDCminus", 1000,0,1000);
  bei->cd();
}


//
//  -- Analyze 
//
void CentralityDQM::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){ 
  using namespace edm;  
  edm::Handle<reco::Centrality> cent;
  iEvent.getByLabel(centrality_, cent); //_centralitytag comes from the cfg as an inputTag and is "hiCentrality"

  if (!cent.isValid()) return;

  h_hiNpix->Fill(cent->multiplicityPixel());
  h_hiNpixelTracks ->Fill( cent->NpixelTracks());
  h_hiNtracks ->Fill( cent->Ntracks()); // 
  h_hiHF ->Fill( cent->EtHFtowerSum()); //
  h_hiHFplus ->Fill( cent->EtHFtowerSumPlus());
  h_hiHFminus ->Fill( cent->EtHFtowerSumMinus());
  h_hiHFplusEta4 ->Fill( cent->EtHFtruncatedPlus());
  h_hiHFminusEta4 ->Fill( cent->EtHFtruncatedMinus());
  h_hiZDC ->Fill( cent->zdcSum());
  h_hiZDCplus ->Fill( cent->zdcSumPlus());
  h_hiZDCminus ->Fill( cent->zdcSumMinus());
  h_hiEEplus ->Fill( cent->EtEESumPlus());
  h_hiEEminus ->Fill( cent->EtEESumMinus());
  h_hiEE ->Fill( cent->EtEESum());
  h_hiEB ->Fill( cent->EtEBSum());
  h_hiET ->Fill( cent->EtMidRapiditySum());
  
}

//
// -- End Luminosity Block
void CentralityDQM::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) { nLumiSecs_++;}

//
// -- End Run
//
void CentralityDQM::endRun(edm::Run const& run, edm::EventSetup const& eSetup){
}



//
// -- End Job
//
void CentralityDQM::endJob(){
  //edm::LogInfo("CentralityDQM") <<"[CentralityDQM]: endjob called!";
}
