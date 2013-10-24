#include "DQM/Physics/src/B2GDQM.h"

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


//
// -- Constructor
//
B2GDQM::B2GDQM(const edm::ParameterSet& ps){

  edm::LogInfo("B2GDQM") <<  " Starting B2GDQM " << "\n" ;


  typedef std::vector<edm::InputTag> vtag;

  // Get parameters from configuration file
  // Trigger
  theTriggerResultsCollection = ps.getParameter<InputTag>("triggerResultsCollection");

  // Jets
  jetLabels_ = ps.getParameter<std::vector<edm::InputTag> >("jetLabels");
  jetPtMins_ = ps.getParameter<std::vector<double> > ("jetPtMins");
  PFJetCorService_ = ps.getParameter<std::string>("PFJetCorService");

  // MET
  PFMETLabel_         = ps.getParameter<InputTag>("pfMETCollection");  




  bei_ = Service<DQMStore>().operator->();
  bei_->setCurrentFolder("Physics/B2G");
  bookHistos(bei_);

}


//
// -- Destructor
//
B2GDQM::~B2GDQM(){
  edm::LogInfo("B2GDQM") <<  " Deleting B2GDQM " << "\n" ;
}


//
// -- Begin Job
//
void B2GDQM::beginJob(){
  nLumiSecs_ = 0;
  nEvents_   = 0;
}


//
// -- Begin Run
//
void B2GDQM::beginRun(Run const& run, edm::EventSetup const& eSetup) {
  edm::LogInfo ("B2GDQM") <<"[B2GDQM]: Begining of Run";
  
  // passed as parameter to HLTConfigProvider::init(), not yet used
  bool isConfigChanged = false;
  
  // isValidHltConfig_ used to short-circuit analyze() in case of problems
  //  const std::string hltProcessName( "HLT" );
  const std::string hltProcessName = theTriggerResultsCollection.process();
  isValidHltConfig_ = hltConfigProvider_.init( run, eSetup, hltProcessName, isConfigChanged );

}


//
// -- Begin  Luminosity Block
//
void B2GDQM::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, 
				  edm::EventSetup const& context) { 
  //edm::LogInfo ("B2GDQM") <<"[B2GDQM]: Begin of LS transition";
}


//
//  -- Book histograms
//
void B2GDQM::bookHistos(DQMStore* bei){
 
  bei->cd();
  
  //--- Event Interpretation

  for ( unsigned int icoll = 0; icoll < jetLabels_.size(); ++icoll ) {
    std::stringstream ss;
    ss << "Physics/B2G/" << jetLabels_[icoll].label();
    bei->setCurrentFolder(ss.str().c_str());
    pfJet_pt          .push_back( bei->book1D("pfJet_pt",     "Pt of PFJet (GeV)",      50, 0.0, 4000) );
    pfJet_y           .push_back( bei->book1D("pfJet_y",      "Rapidity of PFJet",      60, -6.0, 6.0) );
    pfJet_phi         .push_back( bei->book1D("pfJet_phi",    "#phi of PFJet (radians)",50, -3.14159, 3.14159) );
    pfJet_m           .push_back( bei->book1D("pfJet_m",      "Mass of PFJet (GeV)",    50, 0.0, 400) );
    pfJet_chef        .push_back( bei->book1D("pfJet_pfchef", "PFJetID CHEF", 50, 0.0 , 1.0));
    pfJet_nhef        .push_back( bei->book1D("pfJet_pfnhef", "PFJetID NHEF", 50, 0.0 , 1.0));
    pfJet_cemf        .push_back( bei->book1D("pfJet_pfcemf", "PFJetID CEMF", 50, 0.0 , 1.0));
    pfJet_nemf        .push_back( bei->book1D("pfJet_pfnemf", "PFJetID NEMF", 50, 0.0 , 1.0));
  }

  bei->setCurrentFolder("Physics/B2G/MET");
  pfMet_pt                 = bei->book1D("pfMet_pt", "Pf Missing p_{T}; GeV", 50,  0.0 , 500);
  pfMet_phi                = bei->book1D("pfMet_phi", "Pf Missing p_{T} #phi;#phi (radians)", 35, -3.5, 3.5 );


  
  bei->cd();
}


//
//  -- Analyze
//
void B2GDQM::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){

  analyzeEventInterpretation(iEvent, iSetup);
}


void B2GDQM::analyzeEventInterpretation(const Event & iEvent, const edm::EventSetup& iSetup){



  for ( unsigned int icoll = 0; icoll < jetLabels_.size(); ++icoll ) {

    edm::Handle<edm::View<reco::Jet> > pfJetCollection;
    bool ValidPFJets = iEvent.getByLabel(jetLabels_[icoll], pfJetCollection );
    if(!ValidPFJets) continue;
    edm::View<reco::Jet> const & pfjets = *pfJetCollection;
    
    

    // Jet Correction
    int countJet = 0;
    //const JetCorrector* pfcorrector = JetCorrector::getJetCorrector(PFJetCorService_,iSetup);
    
    for ( edm::View<reco::Jet>::const_iterator jet = pfjets.begin(),
	    jetEnd = pfjets.end(), jetBegin = pfjets.begin();
	  jet != jetEnd; ++jet ) {
      if(jet->pt()<jetPtMins_[icoll]) continue;
      pfJet_pt  [icoll]->Fill(jet->pt());
      pfJet_y   [icoll]->Fill(jet->rapidity());
      pfJet_phi [icoll]->Fill(jet->phi());
      pfJet_m   [icoll]->Fill(jet->mass());

      reco::PFJet const * pfjet = dynamic_cast<reco::PFJet const *>( &*jet);

      if ( pfjet != 0 ) {
	pfJet_chef[icoll]->Fill(pfjet->chargedHadronEnergyFraction());
	pfJet_nhef[icoll]->Fill(pfjet->neutralHadronEnergyFraction());
	pfJet_cemf[icoll]->Fill(pfjet->chargedEmEnergyFraction());
	pfJet_nemf[icoll]->Fill(pfjet->neutralEmEnergyFraction());
      }
      countJet++;
    }
  }

  

  // PFMETs
  edm::Handle<std::vector<reco::PFMET> > pfMETCollection;
  bool ValidPFMET = iEvent.getByLabel(PFMETLabel_, pfMETCollection);
  if(!ValidPFMET) return;

  pfMet_pt->Fill( (*pfMETCollection)[0].pt() );
  pfMet_phi->Fill( (*pfMETCollection)[0].phi() );

}

// -- End Luminosity Block
//
void B2GDQM::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) {
  //edm::LogInfo ("B2GDQM") <<"[B2GDQM]: End of LS transition, performing the DQM client operation";
  nLumiSecs_++;
  //edm::LogInfo("B2GDQM") << "============================================ " 
  //<< endl << " ===> Iteration # " << nLumiSecs_ << " " << lumiSeg.luminosityBlock() 
  //<< endl  << "============================================ " << endl;
}


//
// -- End Run
//
void B2GDQM::endRun(edm::Run const& run, edm::EventSetup const& eSetup){
}


//
// -- End Job
//
void B2GDQM::endJob(){
  //edm::LogInfo("B2GDQM") <<"[B2GDQM]: endjob called!";
}



