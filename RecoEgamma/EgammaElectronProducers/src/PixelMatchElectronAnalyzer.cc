// -*- C++ -*-
//
// Package:    ElectronPixelSeed
// Class:      ElectronPixelSeedProducer
// 
/**\class ElectronPixelSeedAnalyzer RecoEgamma/ElectronTrackSeedProducers/src/ElectronPixelSeedAnalyzer.cc

 Description: rereading of electron seeds for verification

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ursula Berthon
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id: PixelMatchElectronAnalyzer.cc,v 1.1 2006/10/03 12:10:17 uberthon Exp $
//
//

// user include files
#include "RecoEgamma/EgammaElectronProducers/interface/PixelMatchElectronAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include <iostream>
#include "TFile.h"
#include "TH1F.h"
#include "TH1I.h"
#include "TTree.h"
#include <iostream>

using namespace reco;
 
PixelMatchElectronAnalyzer::PixelMatchElectronAnalyzer(const edm::ParameterSet& iConfig)
{

  histfile_ = new TFile("electronHistos.root","RECREATE");
}  
  
PixelMatchElectronAnalyzer::~PixelMatchElectronAnalyzer()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  histfile_->Write();
  histfile_->Close();
}

void PixelMatchElectronAnalyzer::beginJob(edm::EventSetup const&iSetup){

  histCharge_= new TH1F("chargeEl","charge, 35 GeV",10, -2.,2.);
  histMass_ = new TH1F("massEl","mass, 35 GeV",100,0.,1.);
  histEn_ = new TH1F("energyEl","energy, 35 GeV",100,0.,1000.);
  histSclEn_ = new TH1F("energySCL","energy, 35 GeV",100,0.,1000.);
  histEt_ = new TH1F("etEl","et, 35 GeV",100,0.,1000.);
  histEta_ = new TH1F("etaEl","eta, 35 GeV",100,-2.5,2.5);
  histPhi_ = new TH1F("phiEl","phi, 35 GeV",100,-3.5,3.5);
  histTrPt_ = new TH1F("ptTr","electron track  pt",100,0.,1000.);
  histTrP_ = new TH1F("pTr","electron track  p",100,0.,1000.);
  histTrEta_ = new TH1F("etaTr","electron track  eta",100,-2.5,2.5);
  histTrPhi_ = new TH1F("phiTr","electron track phi",100,-3.5,3.5);
  histEOP_ =new TH1F("esOpT","Enscl/pTrack",100,-3.5,3.5);
}     

void
PixelMatchElectronAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& iSetup)
{

  // get electrons
  
  edm::Handle<ElectronCollection> electrons;
  e.getByType(electrons); 
  edm::LogInfo("")<<"\n\n =================> Treating event "<<e.id()<<" Number of electrons "<<electrons.product()->size();

  for( ElectronCollection::const_iterator MyS= (*electrons).begin(); MyS != (*electrons).end(); ++MyS) {
    
    histCharge_->Fill((*MyS).charge());
    histMass_->Fill((*MyS).mass());
    histEn_->Fill((*MyS).energy());
    if ((*MyS).et()<150.) histEt_->Fill((*MyS).et());
    histEta_->Fill((*MyS).eta());
    histPhi_->Fill((*MyS).phi());

    // get information about track
    reco::TrackRef tr =(*MyS).track();
    histTrPt_->Fill((*tr).outerPt());
    double pTr=tr->outerP();
    histTrP_->Fill(pTr);
    histTrEta_->Fill((*tr).outerEta());
    histTrPhi_->Fill((*tr).outerPhi());
    // information about SCL
    reco::SuperClusterRef sclRef=(*MyS).superCluster();
    histSclEn_->Fill(sclRef->energy());

    // correlation
    histEOP_->Fill((sclRef->energy())/pTr);
  }
  
}


