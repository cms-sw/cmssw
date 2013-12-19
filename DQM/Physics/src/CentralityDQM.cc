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
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Run.h"
#include "DataFormats/Provenance/interface/EventID.h"

// Candidate handling
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/OverlapChecker.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/CompositeCandidateFwd.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"

// Vertex utilities
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

// Other
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/RefToBase.h"

// Math
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"

// vertexing

// Transient tracks
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

//JetCorrection
#include "JetMETCorrections/Objects/interface/JetCorrector.h"

// ROOT
#include "TLorentzVector.h"
// Centrality
#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
#include "DataFormats/HeavyIonEvent/interface/CentralityProvider.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
// STDLIB
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>

using namespace edm;
using namespace std;
using namespace reco; 
using namespace trigger;

typedef vector<string> vstring;

struct SortCandByDecreasingPt {
  bool operator()( const Candidate &c1, const Candidate &c2) const {
    return c1.pt() > c2.pt();
  }
};


//
// -- Constructor
//
CentralityDQM::CentralityDQM(const edm::ParameterSet& ps){

  edm::LogInfo("CentralityDQM") <<  " Starting CentralityDQM " << "\n" ;

  bei_ = Service<DQMStore>().operator->();
  bei_->setCurrentFolder("Physics/Centrality");
  bookHistos(bei_);

  typedef std::vector<edm::InputTag> vtag;

  //  edm::InputTag _centralitytag;
  // just to initialize
  //isValidHltConfig_ = false;
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
  //  nLumiSecs_ = 0;
  //nEvents_   = 0;
  //pi = 3.14159265;
}


//
// -- Begin Run
//
void CentralityDQM::beginRun(Run const& run, edm::EventSetup const& eSetup) {
  edm::LogInfo ("CentralityDQM") <<"[CentralityDQM]: Begining of Run";
  
  // passed as parameter to HLTConfigProvider::init(), not yet used
  //  bool isConfigChanged = false;
  
  // isValidHltConfig_ used to short-circuit analyze() in case of problems
  //  const std::string hltProcessName( "HLT" );


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
  
  //--- Multijets
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
  edm::Handle<reco::Centrality> centrality_;
  iEvent.getByLabel("hiCentrality",centrality_); //_centralitytag comes from the cfg as an inputTag and is "hiCentrality"


  h_hiNpix->Fill(centrality_->multiplicityPixel());
  h_hiNpixelTracks ->Fill( centrality_->NpixelTracks());
  h_hiNtracks ->Fill( centrality_->Ntracks()); // 
  h_hiHF ->Fill( centrality_->EtHFtowerSum()); //
  h_hiHFplus ->Fill( centrality_->EtHFtowerSumPlus());
  h_hiHFminus ->Fill( centrality_->EtHFtowerSumMinus());
  h_hiHFplusEta4 ->Fill( centrality_->EtHFtruncatedPlus());
  h_hiHFminusEta4 ->Fill( centrality_->EtHFtruncatedMinus());
  h_hiZDC ->Fill( centrality_->zdcSum());
  h_hiZDCplus ->Fill( centrality_->zdcSumPlus());
  h_hiZDCminus ->Fill( centrality_->zdcSumMinus());
  h_hiEEplus ->Fill( centrality_->EtEESumPlus());
  h_hiEEminus ->Fill( centrality_->EtEESumMinus());
  h_hiEE ->Fill( centrality_->EtEESum());
  h_hiEB ->Fill( centrality_->EtEBSum());
  h_hiET ->Fill( centrality_->EtMidRapiditySum());
  
}

void CentralityDQM::analyzeEventInterpretation(const Event & iEvent, const edm::EventSetup& iSetup){  }

//
// -- End Luminosity Block
void CentralityDQM::analyzeMultiJets(const Event & iEvent){}
void CentralityDQM::analyzeMultiJetsTrigger(const Event & iEvent){}
void CentralityDQM::analyzeLongLived(const Event & iEvent){ }
void CentralityDQM::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) { nLumiSecs_++;}
void CentralityDQM::analyzeLongLivedTrigger(const Event & iEvent){}


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
