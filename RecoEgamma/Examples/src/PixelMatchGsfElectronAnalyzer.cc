// -*- C++ -*-
//
// Package:    RecoEgamma/Examples
// Class:      PixelMatchGsfElectronAnalyzer
// 
/**\class PixelMatchGsfElectronAnalyzer RecoEgamma/Examples/src/PixelMatchGsfElectronAnalyzer.cc

 Description: rereading of PixelMatchGsfElectrons for verification

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ursula Berthon
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id: PixelMatchGsfElectronAnalyzer.cc,v 1.5 2007/03/25 11:28:22 futyand Exp $
//
//

// user include files
#include "RecoEgamma/Examples/interface/PixelMatchGsfElectronAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include <iostream>
#include "TFile.h"
#include "TH1F.h"
#include "TH1I.h"
#include "TTree.h"
#include <iostream>

using namespace reco;
 
PixelMatchGsfElectronAnalyzer::PixelMatchGsfElectronAnalyzer(const edm::ParameterSet& conf)
{

  histfile_ = new TFile("gsfElectronHistos.root","RECREATE");
  electronProducer_=conf.getParameter<std::string>("ElectronProducer");
  electronLabel_=conf.getParameter<std::string>("ElectronLabel");
  barrelClusterShapeAssocProducer_ = conf.getParameter<edm::InputTag>("barrelClusterShapeAssociation");
  endcapClusterShapeAssocProducer_ = conf.getParameter<edm::InputTag>("endcapClusterShapeAssociation");
  MCTruthProducer_ = conf.getParameter<std::string>("MCTruthProducer");
}  
  
PixelMatchGsfElectronAnalyzer::~PixelMatchGsfElectronAnalyzer()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  histfile_->Write();
  histfile_->Close();
}

void PixelMatchGsfElectronAnalyzer::beginJob(edm::EventSetup const&iSetup){

  histCharge_= new TH1F("chargeEl","charge, 35 GeV",10, -2.,2.);
  histMass_ = new TH1F("massEl","mass, 35 GeV",20,0.,600.);
  histEn_ = new TH1F("energyEl","energy, 35 GeV",100,0.,100.);
  //histEt_ = new TH1F("etEl","et, 35 GeV",150,0.,15.);
  histEt_ = new TH1F("etEl","et, 35 GeV",150,0.,50.);
  histEta_ = new TH1F("etaEl","eta, 35 GeV",100,-2.5,2.5);
  histPhi_ = new TH1F("phiEl","phi, 35 GeV",100,-3.5,3.5);

  histTrCharge_ = new TH1F("chargeTr","charge of track",10, -2.,2.);
  histTrInP_ = new TH1F("InnerP_Tr","electron track inner p",100,0.,100.);
  //  histTrInPt_ = new TH1F("InnerPt_Tr","electron track inner pt",150,0.,15.);
  histTrInPt_ = new TH1F("InnerPt_Tr","electron track inner pt",150,0.,50.);
  histTrInEta_ = new TH1F("InEtaTr","electron track inner eta",100,-2.5,2.5);
  histTrInPhi_ = new TH1F("InPhiTr","electron track inner phi",100,-3.5,3.5);
  histTrNrHits_=  new TH1F("NrHits","nr hits of electron track",100,0.,25.);
  histTrNrVHits_= new TH1F("NrVHits","nr valid hits of electron track",100,0.,25.);
  histTrChi2_= new TH1F("Chi2","chi2/ndof of electron track",100,0.,100.);
  //  histTrOutPt_ = new TH1F("OuterPt_Tr","electron track outer pt",150,0.,15.);
  histTrOutPt_ = new TH1F("OuterPt_Tr","electron track outer pt",150,0.,50.);
  histTrOutP_ = new TH1F("OuterP_Tr","electron track outer p",100,0.,100.);
 
  histSclEn_ = new TH1F("energySCL","energy, 35 GeV",100,0.,100.);
  //histSclEt_ = new TH1F("etSCL","energy transverse of Supercluster",150,0.,15.);
  histSclEt_ = new TH1F("etSCL","energy transverse of Supercluster",150,0.,50.);
  histSclEta_ = new TH1F("etaSCL","eta of Supercluster",100,-2.5,2.5);
  histSclPhi_ = new TH1F("phiSCL","phi of Supercluster",100,-3.5,3.5);

  histESclOPTr_ =new TH1F("esOpT","Enscl/pTrack",50,0.,5.);
  histDeltaEta_ =new TH1F("DeltaEta","Eta Scl - Eta Track",100,-0.01,0.01);
  histDeltaPhi_ =new TH1F("DeltaPhi","Phi Scl - Phi Track",100,-0.1,0.1);

  histS1overS9_ =new TH1F("S1overS9","ratio of max energy crystal to 3x3 energy for seed BasicCluster",100,0.,1.);

  hist_EOverTruth_ = new TH1F("EOverTruth","Reco energy over true energy",100,.5,1.5);
  hist_EtOverTruth_ = new TH1F("EtOverTruth","Reco Et over True Et",100,.5,1.5);
  hist_DeltaEtaTruth_ = new TH1F("DeltaEtaTruth","Reco eta - true eta",100,-.01,.01);
  hist_DeltaPhiTruth_ = new TH1F("DeltaPhiTruth","Reco phi - true phi",100,-.1,.1);
}     

void
PixelMatchGsfElectronAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // get electrons
  
  edm::Handle<PixelMatchGsfElectronCollection> electrons;
  iEvent.getByLabel(electronProducer_,electronLabel_,electrons); 
  edm::LogInfo("")<<"\n\n =================> Treating event "<<iEvent.id()<<" Number of electrons "<<electrons.product()->size();

  PixelMatchGsfElectronCollection::const_iterator electron;
  for(electron = (*electrons).begin(); electron != (*electrons).end(); ++electron) {
    
    //electron quantities
    histCharge_->Fill((*electron).charge());
    histMass_->Fill((*electron).mass());
    histEn_->Fill((*electron).energy());
    if ((*electron).et()<150.) histEt_->Fill((*electron).et());
    histEta_->Fill((*electron).eta());
    histPhi_->Fill((*electron).phi());

    // track informations 
    reco::GsfTrackRef tr =(*electron).gsfTrack();
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
    reco::SuperClusterRef sclRef=(*electron).superCluster();
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
    histDeltaEta_->Fill((*electron).deltaEtaSuperClusterTrackAtVtx());
    //CC@@
    //histDeltaPhi_->Fill(sclRef->phi()-(*tr).outerPhi());
    histDeltaPhi_->Fill((*electron).deltaPhiSuperClusterTrackAtVtx());

    // Get association maps linking BasicClusters to ClusterShape
    edm::Handle<reco::BasicClusterShapeAssociationCollection> barrelClShpHandle;
    iEvent.getByLabel(barrelClusterShapeAssocProducer_, barrelClShpHandle);
    edm::Handle<reco::BasicClusterShapeAssociationCollection> endcapClShpHandle;
    iEvent.getByLabel(endcapClusterShapeAssocProducer_, endcapClShpHandle);

    reco::BasicClusterShapeAssociationCollection::const_iterator seedShpItr;

    // Find the entry in the map corresponding to the seed BasicCluster of the SuperCluster
    DetId id = sclRef->seed()->getHitsByDetId()[0];
    if (id.subdetId() == EcalBarrel) {
      seedShpItr = barrelClShpHandle->find(sclRef->seed());
    } else {
      seedShpItr = endcapClShpHandle->find(sclRef->seed());
    }

    // Get the ClusterShapeRef corresponding to the BasicCluster
    const reco::ClusterShapeRef& seedShapeRef = seedShpItr->val;
    histS1overS9_->Fill(seedShapeRef->eMax()/seedShapeRef->e3x3());
  }

  // Get generator level information
  edm::Handle<edm::HepMCProduct> hepMCHandle ;
  iEvent.getByLabel(MCTruthProducer_, hepMCHandle) ;
  const HepMC::GenEvent * genEvent = hepMCHandle->GetEvent();

  const double pi = 3.14159;

  // Loop over MC electrons
  HepMC::GenEvent::particle_const_iterator currentParticle; 
  for(currentParticle = genEvent->particles_begin(); 
      currentParticle != genEvent->particles_end(); currentParticle++ ) {
    if(abs((*currentParticle)->pdg_id())==11 && (*currentParticle)->status()==1) {
      double phiTrue = (*currentParticle)->momentum().phi();
      double etaTrue = (*currentParticle)->momentum().eta();
      double eTrue  = (*currentParticle)->momentum().e();
      double etTrue  = (*currentParticle)->momentum().e()/cosh(etaTrue);   

      double etaFound = 0.;
      double phiFound = 0.;
      double etFound = 0.;
      double eFound = 0.;
      double deltaRMin = 999.;
      double deltaPhiMin = 999.;

      // find closest RECO electron to MC electron
      for(electron = electrons->begin(); electron != electrons->end(); electron++) {
	double deltaEta = electron->eta() - etaTrue;
	double deltaPhi = electron->phi() - phiTrue;
	if(deltaPhi > pi) deltaPhi -= 2.*pi;
	if(deltaPhi < -pi) deltaPhi += 2.*pi;
	double deltaR = sqrt(deltaEta*deltaEta + deltaPhi*deltaPhi);

	if(deltaR < deltaRMin) {
	  etFound  = electron->et();
	  eFound   = electron->energy();
	  etaFound = electron->eta();
	  phiFound = electron->phi();
	  deltaRMin = deltaR;
	  deltaPhiMin = deltaPhi;
	}
      }

      // Fill histos for matched electrons
      if(deltaRMin < 0.1) { 
	hist_EtOverTruth_->Fill(etFound/etTrue);   
	hist_EOverTruth_->Fill(eFound/eTrue);
	hist_DeltaEtaTruth_->Fill(etaFound-etaTrue);
	hist_DeltaPhiTruth_->Fill(deltaPhiMin);
      }    
    }
  }
}


