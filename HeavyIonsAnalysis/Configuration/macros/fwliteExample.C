// Usage:
// .x rootlogon.C
// .x fwliteExample.C++

#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <cmath>

#include <TH1D.h>
#include <TH2D.h>
#include <TNtuple.h>
#include <TFile.h>
#include <TROOT.h>
#include <TSystem.h>
#include <TString.h>

#if !defined(__CINT__) && !defined(__MAKECINT__)
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/ChainEvent.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#endif

void fwliteExample(bool debug=false){

  // event cuts
  const unsigned int maxEvents = -1;
  const double hfEThreshold = 3.0;
  const int nTowerThreshold = 1;

  // track cuts
  const double normD0Cut = 5.0;
  const double normDZCut = 5.0;
  const double ptDebug = 3.0;

  // trigger names
  const int nTrigs = 4;
  const char *hltNames[nTrigs] = {"HLT_MinBiasBSC","HLT_L1Jet6U","HLT_Jet15U","HLT_Jet30U"};

  //----- input files (900 GeV data) -----
  vector<string> fileNames;
  fileNames.push_back("./hiCommonSkimAOD.root");
  //fileNames.push_back("../test/hiCommonSkimAOD.root");
  fwlite::ChainEvent event(fileNames);
  
  //----- define output hists/trees in directories of output file -----
  TFile *outFile = new TFile("output_fwlite.root","recreate");
  TH1D::SetDefaultSumw2();

  // evt hists
  outFile->cd(); outFile->mkdir("evt"); outFile->cd("evt");
  TH1D *hL1TechBits = new TH1D("hL1TechBits","L1 technical trigger bits before mask",64,-0.5,63.5);
  TH2D *hHfTowers   = new TH2D("hHfTowers","Number of HF tower above threshold; positive side; negative side",80,-0.5,79.5,80,-0.5,79.5);
  TH1D *hHLTPaths   = new TH1D("hHLTPaths","HLT Paths",3,0,3);
  hHLTPaths->SetCanExtend(TH1::kAllAxes);

  // vtx hists
  outFile->cd(); outFile->mkdir("vtx"); outFile->cd("vtx");
  TH1D *hVtxTrks    = new TH1D("hVtxTrks","number of tracks used to fit pixel vertex",50,-0.5,49.5);
  TH1D *hVtxZ       = new TH1D("hVtxZ","z position of best reconstructed pixel vertex", 80,-20,20);
 
  // track hists
  outFile->cd(); outFile->mkdir("trk"); outFile->cd("trk");
  TH1D *hTrkPt      = new TH1D("hTrkPt","track p_{T}; p_{T} [GeV/c]", 80, 0.0, 20.0);
  TH1D *hTrkEta     = new TH1D("hTrkEta","track #eta; #eta", 60, -3.0, 3.0);
  TH1D *hTrkPhi     = new TH1D("hTrkPhi","track #phi; #phi [radians]", 56, -3.5, 3.5);

  // correlation hists
  outFile->cd(); outFile->mkdir("corr"); outFile->cd("corr");
  TH2D *hDetaDphi   = new TH2D("hDetaDphi","raw two-particle correlation; #Delta#eta; #Delta#phi",50,-5.0,5.0,50,-3.1416,3.1416);

  // debug ntuple
  outFile->cd();
  TNtuple *nt=0;
  if(debug) nt = new TNtuple("nt","track debug ntuple",
			     "pt:eta:phi:hits:pterr:d0:d0err:dz:dzerr:jet6:jet15:jet30");

  //----- loop over events -----
  unsigned int iEvent=0;
  for(event.toBegin(); !event.atEnd(); ++event, ++iEvent){

    if( iEvent == maxEvents ) break;
    if( iEvent % 1000 == 0 ) cout << "Processing " << iEvent<< "th event: "
				  << "run " << event.id().run() 
				  << ", lumi " << event.luminosityBlock() 
				  << ", evt " << event.id().event() << endl;

    // select on L1 trigger bits
    fwlite::Handle<L1GlobalTriggerReadoutRecord> gt;
    gt.getByLabel(event, "gtDigis");
    const TechnicalTriggerWord&  word = gt->technicalTriggerWord(); //before mask
    for(int bit=0; bit<64; bit++) hL1TechBits->Fill(bit,word.at(bit));
    if(!word.at(0)) continue;  // BPTX coincidence
    if(!(word.at(40) || word.at(41))) continue; // BSC coincidence
    if(word.at(36) || word.at(37) || word.at(38) || word.at(39)) continue; // BSC halo
    
    // select on coincidence of HF towers above threshold
    fwlite::Handle<CaloTowerCollection> towers;
    towers.getByLabel(event, "towerMaker");
    int nHfTowersN=0, nHfTowersP=0;
    for(CaloTowerCollection::const_iterator calo = towers->begin(); calo != towers->end(); ++calo) {
      if(calo->energy() < hfEThreshold) continue;
      if(calo->eta()>3) nHfTowersP++;
      if(calo->eta()<-3) nHfTowersN++;
    }
    hHfTowers->Fill(nHfTowersP,nHfTowersN);
    if(nHfTowersP < nTowerThreshold || nHfTowersN < nTowerThreshold) continue;
    

    // get hlt bits
    bool accept[nTrigs]={};
    fwlite::Handle<edm::TriggerResults> triggerResults;
    triggerResults.getByLabel(event, "TriggerResults","","HLT");
    const edm::TriggerNames triggerNames = event.triggerNames(*triggerResults);
    for(int i=0; i<nTrigs; i++) {
      accept[i] = triggerResults->accept(triggerNames.triggerIndex(hltNames[i]));
      if(accept[i]) hHLTPaths->Fill(hltNames[i],1);
    }

    // select on requirement of valid vertex
    math::XYZPoint vtxpoint(0,0,0);
    fwlite::Handle<std::vector<reco::Vertex> > vertices;
    vertices.getByLabel(event, "hiSelectedVertex");
    if(!vertices->size()) continue;
    const reco::Vertex & vtx = (*vertices)[0];
    vtxpoint.SetCoordinates(vtx.x(),vtx.y(),vtx.z());
    hVtxTrks->Fill(vtx.tracksSize());
    hVtxZ->Fill(vtx.z());

    // get beamspot 
    fwlite::Handle<reco::BeamSpot> beamspot;
    beamspot.getByLabel(event, "offlineBeamSpot");

    //----- loop over tracks -----

    fwlite::Handle<std::vector<reco::Track> > tracks;
    tracks.getByLabel(event, "hiSelectedTracks");

    for(unsigned it=0; it<tracks->size(); ++it){
      
      const reco::Track & trk = (*tracks)[it];

      // select tracks based on transverse proximity to beamspot
      double dxybeam = trk.dxy(beamspot->position());
      if(fabs(dxybeam/trk.d0Error()) > normD0Cut) continue;

      // select tracks based on z-proximity to best vertex 
      double dzvtx = trk.dz(vtxpoint);
      if(fabs(dzvtx/trk.dzError()) > normDZCut) continue;

      // fill selected track histograms and debug ntuple
      hTrkPt->Fill(trk.pt());
      hTrkEta->Fill(trk.eta());
      hTrkPhi->Fill(trk.phi());
      if(debug && trk.pt() > ptDebug) // fill debug ntuple for selection of tracks
	nt->Fill(trk.pt(),trk.eta(),trk.phi(),trk.numberOfValidHits(),trk.ptError(),
		 dxybeam,trk.d0Error(),dzvtx,trk.dzError(),accept[1],accept[2],accept[3]);
    
    }

    //----- loop over jets -----

    fwlite::Handle<vector<reco::CaloJet> > jets;
    jets.getByLabel(event, "iterativeConePu5CaloJets");

    //----- loop over muons -----

    //----- loop over photons -----

    //----- loop over charged candidates -----
    
    fwlite::Handle<vector<reco::RecoChargedCandidate> > candidates;
    candidates.getByLabel(event, "allTracks");
    
    for(unsigned it1=0; it1<candidates->size(); ++it1) {
      const reco::RecoChargedCandidate & c1 = (*candidates)[it1];
      for(unsigned it2=0; it2<candidates->size(); ++it2) {
	if(it1==it2) continue;
	const reco::RecoChargedCandidate & c2 = (*candidates)[it2];	
	hDetaDphi->Fill(c1.eta()-c2.eta(),deltaPhi(c1.phi(),c2.phi()));
      }
    }

  } //end loop over events
  
  cout << "Number of events processed : " << iEvent << endl;
  cout << "Number passing all event selection cuts: " << hVtxZ->GetEntries() << endl;

  // write to output file
  hHLTPaths->LabelsDeflate();
  outFile->Write();
  outFile->ls();
  outFile->Close();

}
