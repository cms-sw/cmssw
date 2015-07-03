/* This Class Header */
#include "DQMOffline/Muon/interface/EfficiencyAnalyzer.h"

/* Collaborating Class Header */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

#include "TLorentzVector.h"
#include "TFile.h"
#include <vector>
#include <cmath>
#include <algorithm>

/* C++ Headers */
#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;
using namespace edm;

EfficiencyAnalyzer::EfficiencyAnalyzer(const edm::ParameterSet& pSet){
  parameters = pSet;
  
  theService = new MuonServiceProxy(parameters.getParameter<ParameterSet>("ServiceParameters"));

  // DATA 
  theMuonCollectionLabel_  = consumes<reco::MuonCollection>  (parameters.getParameter<edm::InputTag>("MuonCollection"));
  theTrackCollectionLabel_ = consumes<reco::TrackCollection> (parameters.getParameter<edm::InputTag>("TrackCollection"));
  theVertexLabel_          = consumes<reco::VertexCollection>(parameters.getParameter<edm::InputTag>("VertexLabel"));
  theBeamSpotLabel_        = mayConsume<reco::BeamSpot>      (parameters.getParameter<edm::InputTag>("BeamSpotLabel"));

  //Vertex requirements
  doPVCheck_ = parameters.getParameter<bool>("doPrimaryVertexCheck");

  ptBin_ = parameters.getParameter<int>("ptBin");
  ptMin_ = parameters.getParameter<double>("ptMin");
  ptMax_ = parameters.getParameter<double>("ptMax");

  etaBin_ = parameters.getParameter<int>("etaBin");
  etaMin_ = parameters.getParameter<double>("etaMin");
  etaMax_ = parameters.getParameter<double>("etaMax");

  phiBin_ = parameters.getParameter<int>("phiBin");
  phiMin_ = parameters.getParameter<double>("phiMin");
  phiMax_ = parameters.getParameter<double>("phiMax");

  vtxBin_ = parameters.getParameter<int>("vtxBin");
  vtxMin_ = parameters.getParameter<double>("vtxMin");
  vtxMax_ = parameters.getParameter<double>("vtxMax");
  
  ID_     = parameters.getParameter<string>("ID");
}

EfficiencyAnalyzer::~EfficiencyAnalyzer() {
  delete theService;
}

void EfficiencyAnalyzer::bookHistograms(DQMStore::IBooker & ibooker,
					edm::Run const & /*iRun*/,
					edm::EventSetup const & /* iSetup */){
  
  ibooker.cd();
  ibooker.setCurrentFolder("Muons/EfficiencyAnalyzer/"+ID_);  

  h_allProbes_pt     = ibooker.book1D("allProbes_pt","All Probes Pt",              ptBin_, ptMin_, ptMax_);
  h_allProbes_EB_pt  = ibooker.book1D("allProbes_EB_pt","Barrel: all Probes Pt",   ptBin_, ptMin_, ptMax_);
  h_allProbes_EE_pt  = ibooker.book1D("allProbes_EE_pt","Endcap: all Probes Pt",   ptBin_, ptMin_, ptMax_);
  h_allProbes_eta    = ibooker.book1D("allProbes_eta","All Probes Eta",            etaBin_, etaMin_, etaMax_);
  h_allProbes_hp_eta = ibooker.book1D("allProbes_hp_eta","High Pt all Probes Eta", etaBin_, etaMin_, etaMax_);
  h_allProbes_phi    = ibooker.book1D("allProbes_phi","All Probes Phi",            phiBin_, phiMin_, phiMax_);
  
  h_allProbes_ID_pt      = ibooker.book1D("allProbes_ID_pt","All ID Probes Pt", ptBin_, ptMin_, ptMax_);
  h_allProbes_EB_ID_pt   = ibooker.book1D("allProbes_EB_ID_pt","Barrel: all ID Probes Pt", ptBin_, ptMin_, ptMax_);
  h_allProbes_EE_ID_pt   = ibooker.book1D("allProbes_EE_ID_pt","Endcap: all ID Probes Pt", ptBin_, ptMin_, ptMax_);
  h_allProbes_ID_nVtx    = ibooker.book1D("allProbes_ID_nVtx","All Probes (ID) nVtx", vtxBin_, vtxMin_, vtxMax_);
  h_allProbes_EB_ID_nVtx = ibooker.book1D("allProbes_EB_ID_nVtx","Barrel: All Probes (ID) nVtx", vtxBin_, vtxMin_, vtxMax_);
  h_allProbes_EE_ID_nVtx = ibooker.book1D("allProbes_EE_ID_nVtx","Endcap: All Probes (ID) nVtx", vtxBin_, vtxMin_, vtxMax_);

  h_passProbes_ID_pt     = ibooker.book1D("passProbes_ID_pt","ID Passing Probes Pt", ptBin_ , ptMin_ , ptMax_ );
  h_passProbes_ID_EB_pt  = ibooker.book1D("passProbes_ID_EB_pt","Barrel: ID Passing Probes Pt", ptBin_ , ptMin_ , ptMax_ );
  h_passProbes_ID_EE_pt  = ibooker.book1D("passProbes_ID_EE_pt","Endcap: ID Passing Probes Pt", ptBin_ , ptMin_ , ptMax_ );
  h_passProbes_ID_eta    = ibooker.book1D("passProbes_ID_eta","ID Passing Probes #eta", etaBin_, etaMin_, etaMax_);
  h_passProbes_ID_hp_eta = ibooker.book1D("passProbes_ID_hp_eta","High Pt ID Passing Probes #eta", etaBin_, etaMin_, etaMax_);
  h_passProbes_ID_phi    = ibooker.book1D("passProbes_ID_phi","ID Passing Probes #phi", phiBin_, phiMin_, phiMax_);

  h_passProbes_detIsoID_pt = ibooker.book1D("passProbes_detIsoID_pt","detIsoID Passing Probes Pt", ptBin_, ptMin_, ptMax_);
  h_passProbes_EB_detIsoID_pt = ibooker.book1D("passProbes_EB_detIsoID_pt","Barrel: detIsoID Passing Probes Pt", ptBin_, ptMin_, ptMax_);
  h_passProbes_EE_detIsoID_pt = ibooker.book1D("passProbes_EE_detIsoID_pt","Endcap: detIsoID Passing Probes Pt", ptBin_, ptMin_, ptMax_);

  h_passProbes_pfIsoID_pt = ibooker.book1D("passProbes_pfIsoID_pt","pfIsoID Passing Probes Pt", ptBin_, ptMin_, ptMax_);
  h_passProbes_EB_pfIsoID_pt = ibooker.book1D("passProbes_EB_pfIsoID_pt","Barrel: pfIsoID Passing Probes Pt", ptBin_, ptMin_, ptMax_);
  h_passProbes_EE_pfIsoID_pt = ibooker.book1D("passProbes_EE_pfIsoID_pt","Endcap: pfIsoID Passing Probes Pt", ptBin_, ptMin_, ptMax_);

  h_passProbes_detIsoID_nVtx    = ibooker.book1D("passProbes_detIsoID_nVtx",    "detIsoID Passing Probes nVtx (R03)",  vtxBin_, vtxMin_, vtxMax_);
  h_passProbes_pfIsoID_nVtx     = ibooker.book1D("passProbes_pfIsoID_nVtx",    "pfIsoID Passing Probes nVtx (R04)",  vtxBin_, vtxMin_, vtxMax_);
  h_passProbes_EB_detIsoID_nVtx = ibooker.book1D("passProbes_EB_detIsoID_nVtx","Barrel: detIsoID Passing Probes nVtx (R03)",  vtxBin_, vtxMin_, vtxMax_);
  h_passProbes_EE_detIsoID_nVtx = ibooker.book1D("passProbes_EE_detIsoID_nVtx","Endcap: detIsoID Passing Probes nVtx (R03)",  vtxBin_, vtxMin_, vtxMax_);
  h_passProbes_EB_pfIsoID_nVtx  = ibooker.book1D("passProbes_EB_pfIsoID_nVtx", "Barrel: pfIsoID Passing Probes nVtx (R04)",  vtxBin_, vtxMin_, vtxMax_);
  h_passProbes_EE_pfIsoID_nVtx  = ibooker.book1D("passProbes_EE_pfIsoID_nVtx", "Endcap: pfIsoID Passing Probes nVtx (R04)",  vtxBin_, vtxMin_, vtxMax_);

  
  // Apply deltaBeta PU corrections to the PF isolation eficiencies.
  
  h_passProbes_pfIsodBID_pt = ibooker.book1D("passProbes_pfIsodBID_pt","pfIsoID Passing Probes Pt (deltaB PU correction)", ptBin_, ptMin_, ptMax_);
  h_passProbes_EB_pfIsodBID_pt = ibooker.book1D("passProbes_EB_pfIsodBID_pt","Barrel: pfIsoID Passing Probes Pt (deltaB PU correction)", ptBin_, ptMin_, ptMax_);
  h_passProbes_EE_pfIsodBID_pt = ibooker.book1D("passProbes_EE_pfIsodBID_pt","Endcap: pfIsoID Passing Probes Pt (deltaB PU correction)", ptBin_, ptMin_, ptMax_);
  h_passProbes_pfIsodBID_nVtx     = ibooker.book1D("passProbes_pfIsodBID_nVtx",    "pfIsoID Passing Probes nVtx (R04) (deltaB PU correction)",  vtxBin_, vtxMin_, vtxMax_);
h_passProbes_EB_pfIsodBID_nVtx  = ibooker.book1D("passProbes_EB_pfIsodBID_nVtx", "Barrel: pfIsoID Passing Probes nVtx (R04) (deltaB PU correction)",  vtxBin_, vtxMin_, vtxMax_);
  h_passProbes_EE_pfIsodBID_nVtx  = ibooker.book1D("passProbes_EE_pfIsodBID_nVtx", "Endcap: pfIsoID Passing Probes nVtx (R04) (deltaB PU correction)",  vtxBin_, vtxMin_, vtxMax_);

#ifdef DEBUG
  cout << "[EfficiencyAnalyzer] Parameters initialization DONE" <<endl;
#endif
}

void EfficiencyAnalyzer::analyze(const edm::Event & iEvent,const edm::EventSetup& iSetup){

  LogTrace(metname)<<"[EfficiencyAnalyzer] Analyze the mu in different eta regions";
  theService->update(iSetup);
  // ==========================================================
  // BEGIN READ DATA:
  // Muon information
  edm::Handle<reco::MuonCollection> muons;
  iEvent.getByToken(theMuonCollectionLabel_, muons);

  // Tracks information
  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByToken(theTrackCollectionLabel_,tracks); /// to be read from output as "generalTracks"

  //Vertex information
  edm::Handle<reco::VertexCollection> vertex;
  iEvent.getByToken(theVertexLabel_, vertex);  
  // END READ DATA
  // ==========================================================
  

  _numPV = 0;
  bool bPrimaryVertex = true;
  if(doPVCheck_){ 
    bPrimaryVertex = false;
         
    if (!vertex.isValid()) {
      LogTrace(metname) << "[EfficiencyAnalyzer] Could not find vertex collection" << std::endl;
      bPrimaryVertex = false;
    }
    
    if ( vertex.isValid() ){
      const reco::VertexCollection& vertexCollection = *(vertex.product());
      int vertex_number     = vertexCollection.size();
      
      reco::VertexCollection::const_iterator v = vertexCollection.begin();
      for ( ; v != vertexCollection.end(); ++v) {
	double vertex_chi2    = v->normalizedChi2();
	double vertex_ndof    = v->ndof();
	bool   fakeVtx        = v->isFake();
	double vertex_Z       = v->z();
	
	if (  !fakeVtx
	      && vertex_number >= 1
	      && vertex_ndof   > 4
	      && vertex_chi2   < 999
	      && fabs(vertex_Z)< 24. ) {
	  bPrimaryVertex = true;
	  ++_numPV;
	}
      }
    }
  
  }
  
  // =================================================================================
  // Look for the Primary Vertex (and use the BeamSpot instead, if you can't find it):
  reco::Vertex::Point posVtx;
  reco::Vertex::Error errVtx;
  unsigned int theIndexOfThePrimaryVertex = 999.;
  if (vertex.isValid()){
    for (unsigned int ind=0; ind<vertex->size(); ++ind) {
      if ( (*vertex)[ind].isValid() && !((*vertex)[ind].isFake()) ) {
	theIndexOfThePrimaryVertex = ind;
	break;
      }
    }
  }
  
  if (theIndexOfThePrimaryVertex<100) {
    posVtx = ((*vertex)[theIndexOfThePrimaryVertex]).position();
    errVtx = ((*vertex)[theIndexOfThePrimaryVertex]).error();
  }   
  else {
    LogInfo("RecoMuonValidator") << "reco::PrimaryVertex not found, use BeamSpot position instead\n";
    
    edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
    iEvent.getByToken(theBeamSpotLabel_,recoBeamSpotHandle);
    reco::BeamSpot bs = *recoBeamSpotHandle;
    
    posVtx = bs.position();
    errVtx(0,0) = bs.BeamWidthX();
    errVtx(1,1) = bs.BeamWidthY();
    errVtx(2,2) = bs.sigmaZ();
  }
    
  const reco::Vertex thePrimaryVertex(posVtx,errVtx);
  // ==========================================================
  
  if(!muons.isValid()) return;
  
  // Loop on muon collection
  TLorentzVector Mu1, Mu2;

  bool isMB = false;
  bool isME = false;

  for (reco::MuonCollection::const_iterator recoMu1 = muons->begin(); recoMu1!=muons->end(); ++recoMu1) {
    LogTrace(metname)<<"[EfficiencyAnalyzer] loop over first muons" << endl;
    
    //--- Define combined isolation
    reco::MuonIsolation Iso_muon = recoMu1->isolationR03();
    float combIso = (Iso_muon.emEt + Iso_muon.hadEt + Iso_muon.sumPt);  
    
    //--- Is Global Muon 
    if (!recoMu1->isGlobalMuon()) continue;

    // get the track combinig the information from both the Tracker and the Spectrometer
    reco::TrackRef recoCombinedGlbTrack1 = recoMu1->combinedMuon();    
    float muPt1 = recoCombinedGlbTrack1->pt();
    Mu1.SetPxPyPzE(recoCombinedGlbTrack1->px(), recoCombinedGlbTrack1->py(),recoCombinedGlbTrack1->pz(), recoCombinedGlbTrack1->p());
    

    //--- Define if it is a tight muon
    // Change the Tight muon definition by using the implemented method in: MuonSelectors.cc
    if (ID_ == "Loose"  && !muon::isLooseMuon(*recoMu1)) continue; 
    if (ID_ == "Medium" && !muon::isMediumMuon(*recoMu1)) continue;
    if (ID_ == "Tight"  && !muon::isTightMuon(*recoMu1, thePrimaryVertex)) continue;
  
    //-- is isolated muon
    if (muPt1 <= 15)          continue;
    if (combIso/muPt1 > 0.1 ) continue;
    
    for (reco::MuonCollection::const_iterator recoMu2 = muons->begin(); recoMu2!=muons->end(); ++recoMu2){ 
      LogTrace(metname)<<"[EfficiencyAnalyzer] loop over second muon" <<endl;
      if (recoMu2 == recoMu1) continue;
      
      if (recoMu2->eta() < 1.479 )  isMB = true;
      if (recoMu2->eta() >= 1.479 ) isME = true;
      
      //--> should we apply track quality cuts??? 
      Mu2.SetPxPyPzE(recoMu2->px(), recoMu2->py(), recoMu2->pz(), recoMu2->p());
      
      float Minv = (Mu1+Mu2).M();
      if (!recoMu2->isTrackerMuon())                     continue;
      if ( recoMu2->pt() < 5 )                           continue;
      if ( (recoMu1->charge())*(recoMu2->charge()) > 0 ) continue; 
      if ( Minv < 70 ||  Minv > 110 )                    continue;
      
      h_allProbes_pt->Fill(recoMu2->pt());
      h_allProbes_eta->Fill(recoMu2->eta());
      h_allProbes_phi->Fill(recoMu2->phi());
      
      if (isMB)               h_allProbes_EB_pt->Fill(recoMu2->pt());
      if (isME)               h_allProbes_EE_pt->Fill(recoMu2->pt());
      if(recoMu2->pt() > 20 ) h_allProbes_hp_eta->Fill(recoMu2->eta());

            
      // Probes passing the tight muon criteria 
      if (ID_ == "Loose"  && !muon::isLooseMuon(*recoMu2)) continue; 
      if (ID_ == "Medium" && !muon::isMediumMuon(*recoMu2)) continue;
      if (ID_ == "Tight"  && !muon::isTightMuon(*recoMu2, thePrimaryVertex)) continue;


      h_passProbes_ID_pt->Fill(recoMu2->pt());
      h_passProbes_ID_eta->Fill(recoMu2->eta());
      h_passProbes_ID_phi->Fill(recoMu2->phi());
      
      if (isMB) h_passProbes_ID_EB_pt->Fill(recoMu2->pt());
      if (isME) h_passProbes_ID_EE_pt->Fill(recoMu2->pt());
      if( recoMu2->pt() > 20 ) h_passProbes_ID_hp_eta->Fill(recoMu2->eta());
      
      h_allProbes_ID_pt->Fill(recoMu2->pt());
      if (isMB) h_allProbes_EB_ID_pt->Fill(recoMu2->pt());
      if (isME) h_allProbes_EE_ID_pt->Fill(recoMu2->pt());
      
      //------- For PU monitoring -------//
      if (bPrimaryVertex)         h_allProbes_ID_nVtx->Fill(_numPV);
      if (bPrimaryVertex && isMB) h_allProbes_EB_ID_nVtx->Fill(_numPV);
      if (bPrimaryVertex && isME) h_allProbes_EE_ID_nVtx->Fill(_numPV);
      
      //-- Define det relative isolation
      float tkIso = recoMu2->isolationR03().sumPt;
      float emIso = recoMu2->isolationR03().emEt;
      float hadIso = recoMu2->isolationR03().hadEt + recoMu2->isolationR03().hoEt;
      float relDetIso = (tkIso + emIso + hadIso) /  (recoMu2->pt()); 
      
      if (relDetIso < 0.05 ) { 
	h_passProbes_detIsoID_pt->Fill(recoMu2->pt());
	if (isMB) h_passProbes_EB_detIsoID_pt->Fill(recoMu2->pt());
	if (isME) h_passProbes_EE_detIsoID_pt->Fill(recoMu2->pt());
	
	if (bPrimaryVertex)         h_passProbes_detIsoID_nVtx->Fill(_numPV);
	if (bPrimaryVertex && isMB) h_passProbes_EB_detIsoID_nVtx->Fill(_numPV);
	if (bPrimaryVertex && isME) h_passProbes_EE_detIsoID_nVtx->Fill(_numPV);
      }
      
      //-- Define PF relative isolation
      float chargedIso = recoMu2->pfIsolationR04().sumChargedHadronPt;
      float neutralIso = recoMu2->pfIsolationR04().sumNeutralHadronEt;
      float photonIso = recoMu2->pfIsolationR04().sumPhotonEt;
      float relPFIso = (chargedIso + neutralIso + photonIso) /  (recoMu2->pt()); 
      
      float pu = recoMu2->pfIsolationR04().sumPUPt; 
      float neutralphotonPUCorrected = std::max(0.0 , (neutralIso + photonIso - 0.5*pu ) ); 
      float relPFIsoPUCorrected = (chargedIso + neutralphotonPUCorrected) /  (recoMu2->pt()); 
      
	
      
      if (relPFIso < 0.12 ) { 
	h_passProbes_pfIsoID_pt->Fill(recoMu2->pt());
	if (isMB) h_passProbes_EB_pfIsoID_pt->Fill(recoMu2->pt());
	if (isME) h_passProbes_EE_pfIsoID_pt->Fill(recoMu2->pt());
	
	if( bPrimaryVertex)         h_passProbes_pfIsoID_nVtx->Fill(_numPV);
	if (bPrimaryVertex && isMB) h_passProbes_EB_pfIsoID_nVtx->Fill(_numPV);
	if (bPrimaryVertex && isME) h_passProbes_EE_pfIsoID_nVtx->Fill(_numPV);
      }
      
      // Apply deltaBeta PU corrections to the PF isolation eficiencies.
      if ( relPFIsoPUCorrected < 0.12 ) { 
	h_passProbes_pfIsodBID_pt->Fill(recoMu2->pt());
	if (isMB) h_passProbes_EB_pfIsodBID_pt->Fill(recoMu2->pt());
	if (isME) h_passProbes_EE_pfIsodBID_pt->Fill(recoMu2->pt());
	
	if( bPrimaryVertex)         h_passProbes_pfIsodBID_nVtx->Fill(_numPV);
	if (bPrimaryVertex && isMB) h_passProbes_EB_pfIsodBID_nVtx->Fill(_numPV);
	if (bPrimaryVertex && isME) h_passProbes_EE_pfIsodBID_nVtx->Fill(_numPV);
      }
    }
  }
}


