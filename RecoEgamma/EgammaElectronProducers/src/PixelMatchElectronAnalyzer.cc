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
// $Id: PixelMatchElectronAnalyzer.cc,v 1.8 2006/12/21 16:12:54 uberthon Exp $
//
//

// user include files
#include "RecoEgamma/EgammaElectronProducers/interface/PixelMatchElectronAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EgammaCandidates/interface/PixelMatchElectron.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include <iostream>
#include "TFile.h"
#include "TH1F.h"
#include "TH1I.h"
#include "TTree.h"
#include <iostream>

using namespace reco;
 
PixelMatchElectronAnalyzer::PixelMatchElectronAnalyzer(const edm::ParameterSet& conf)
{

  histfile_ = new TFile("electronHistos.root","RECREATE");
  electronProducer_=conf.getParameter<std::string>("ElectronProducer");
  electronLabel_=conf.getParameter<std::string>("ElectronLabel");
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
  histMass_ = new TH1F("massEl","mass, 35 GeV",20,0.,600.);
  histEn_ = new TH1F("energyEl","energy, 35 GeV",100,0.,100.);
  histEt_ = new TH1F("etEl","et, 35 GeV",150,0.,15.);
  histEta_ = new TH1F("etaEl","eta, 35 GeV",100,-2.5,2.5);
  histPhi_ = new TH1F("phiEl","phi, 35 GeV",100,-3.5,3.5);

  histTrCharge_ = new TH1F("chargeTr","charge of track",10, -2.,2.);
  histTrInP_ = new TH1F("InnerP_Tr","electron track inner p",100,0.,100.);
  histTrInPt_ = new TH1F("InnerPt_Tr","electron track inner pt",150,0.,15.);
  histTrInEta_ = new TH1F("InEtaTr","electron track inner eta",100,-2.5,2.5);
  histTrInPhi_ = new TH1F("InPhiTr","electron track inner phi",100,-3.5,3.5);
  histTrNrHits_=  new TH1F("NrHits","nr hits of electron track",100,0.,25.);
  histTrNrVHits_= new TH1F("NrVHits","nr valid hits of electron track",100,0.,25.);
  histTrChi2_= new TH1F("Chi2","chi2/ndof of electron track",100,0.,100.);
  histTrOutPt_ = new TH1F("OuterPt_Tr","electron track outer pt",150,0.,15.);
  histTrOutP_ = new TH1F("OuterP_Tr","electron track outer p",100,0.,100.);
 
  histSclEn_ = new TH1F("energySCL","energy, 35 GeV",100,0.,100.);
  histSclEt_ = new TH1F("etSCL","energy transverse of Supercluster",150,0.,15.);
  histSclEta_ = new TH1F("etaSCL","eta of Supercluster",100,-2.5,2.5);
  histSclPhi_ = new TH1F("phiSCL","phi of Supercluster",100,-3.5,3.5);

  histESclOPTr_ =new TH1F("esOpT","Enscl/pTrack",50,0.,5.);
  histDeltaEta_ =new TH1F("DeltaEta","Eta Scl - Eta Track",100,-0.25,0.25);
  histDeltaPhi_ =new TH1F("DeltaPhi","Phi Scl - Phi Track",100,-0.25,0.25);
}     

void
PixelMatchElectronAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& iSetup)
{

  // get electrons
  
  edm::Handle<PixelMatchElectronCollection> electrons;
  e.getByLabel(electronProducer_,electronLabel_,electrons); 
  edm::LogInfo("")<<"\n\n =================> Treating event "<<e.id()<<" Number of electrons "<<electrons.product()->size();

  for( PixelMatchElectronCollection::const_iterator MyS= (*electrons).begin(); MyS != (*electrons).end(); ++MyS) {
    
    //electron quantities
    histCharge_->Fill((*MyS).charge());
    histMass_->Fill((*MyS).mass());
    histEn_->Fill((*MyS).energy());
    if ((*MyS).et()<150.) histEt_->Fill((*MyS).et());
    histEta_->Fill((*MyS).eta());
    histPhi_->Fill((*MyS).phi());

    // track informations 
    reco::TrackRef tr =(*MyS).track();
    histTrCharge_->Fill(tr->charge());
    histTrInP_->Fill((*tr).innerMomentum().R());
    histTrInPt_->Fill((*tr).innerMomentum().Rho());
    histTrInEta_->Fill((*tr).outerEta());
    histTrInPhi_->Fill((*tr).outerPhi());
    histTrNrHits_->Fill((*tr).recHitsSize());
    histTrNrVHits_->Fill((*tr).found());
    histTrChi2_->Fill((*tr).chi2()/(*tr).ndof());
    double pTr=tr->outerP();
    histTrOutP_->Fill(pTr);
    histTrOutPt_->Fill(tr->outerPt());

    // SCL informations
    reco::SuperClusterRef sclRef=(*MyS).superCluster();
    histSclEn_->Fill(sclRef->energy());
    double R=TMath::Sqrt(sclRef->x()*sclRef->x() + sclRef->y()*sclRef->y() +sclRef->z()*sclRef->z());
    double Rt=TMath::Sqrt(sclRef->x()*sclRef->x() + sclRef->y()*sclRef->y());
    histSclEt_->Fill(sclRef->energy()*(Rt/R));
    histSclEta_->Fill(sclRef->eta());
    histSclPhi_->Fill(sclRef->phi());

    // correlation etc
    histESclOPTr_->Fill((sclRef->energy())/pTr);
    //CC@@
    //histDeltaEta_->Fill(sclRef->eta()-(*tr).outerEta());
    histDeltaEta_->Fill((*MyS).deltaEtaSuperClusterTrackAtVtx());
    //CC@@
    //histDeltaPhi_->Fill(sclRef->phi()-(*tr).outerPhi());
    histDeltaPhi_->Fill((*MyS).deltaPhiSuperClusterTrackAtVtx());

  }
  
}


