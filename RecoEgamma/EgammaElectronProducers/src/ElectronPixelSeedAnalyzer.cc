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
// $Id: ElectronPixelSeedAnalyzer.cc,v 1.7 2006/10/13 16:01:45 uberthon Exp $
//
//

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EgammaReco/interface/ElectronPixelSeedFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoEgamma/EgammaElectronProducers/interface/ElectronPixelSeedAnalyzer.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include <iostream>
#include "TFile.h"
#include "TH1F.h"
#include "TH1I.h"
#include "TTree.h"

using namespace std;
using namespace reco;
 
ElectronPixelSeedAnalyzer::ElectronPixelSeedAnalyzer(const edm::ParameterSet& conf)
{
  histfile_ = new TFile("electronpixelseeds.root","RECREATE");
  
  seedProducer_=conf.getParameter<string>("SeedProducer");
  seedLabel_=conf.getParameter<string>("SeedLabel");
}  
  
ElectronPixelSeedAnalyzer::~ElectronPixelSeedAnalyzer()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  tree_->Print();
  histfile_->Write();
  histfile_->Close();
}

void ElectronPixelSeedAnalyzer::beginJob(edm::EventSetup const&iSetup){
  iSetup.get<TrackerDigiGeometryRecord> ().get (pDD); 
  iSetup.get<IdealMagneticFieldRecord>().get(theMagField);
  tree_ = new TTree("ElectronPixelSeeds","ElectronPixelSeed validation ntuple");
  tree_->Branch("mcEnergy",mcEnergy,"mcEnergy[10]/F");
  tree_->Branch("mcEta",mcEta,"mcEta[10]/F");
  tree_->Branch("mcPhi",mcPhi,"mcPhi[10]/F");
  tree_->Branch("mcPt",mcPt,"mcPt[10]/F");
  tree_->Branch("mcQ",mcQ,"mcQ[10]/F");
  tree_->Branch("superclusterEnergy",superclusterEnergy,"superclusterEnergy[10]/F");
  tree_->Branch("superclusterEta",superclusterEta,"superclusterEta[10]/F");
  tree_->Branch("superclusterPhi",superclusterPhi,"superclusterPhi[10]/F");
  tree_->Branch("superclusterEt",superclusterEt,"superclusterEt[10]/F");
  tree_->Branch("seedMomentum",seedMomentum,"seedMomentum[10]/F");
  tree_->Branch("seedEta",seedEta,"seedEta[10]/F");
  tree_->Branch("seedPhi",seedPhi,"seedPhi[10]/F");
  tree_->Branch("seedPt",seedPt,"seedPt[10]/F");
  tree_->Branch("seedQ",seedQ,"seedQ[10]/F");
  histeMC_ = new TH1F("eMC","MC particle energy",100,0.,100.);
  histp_ = new TH1F("p","seed p",100,0.,100.);
  histeclu_ = new TH1F("clus energy","supercluster energy",100,0.,100.);
  histpt_ = new TH1F("pt","seed pt",100,0.,100.);
  histptMC_ = new TH1F("ptMC","MC particle pt",100,0.,100.);
  histetclu_ = new TH1F("Et","supercluster Et",100,0.,100.);
  histeffpt_ = new TH1F("pt eff","seed effciency vs pt",100,0.,100.);
  histeta_ = new TH1F("seed eta","seed eta",100,-2.5,2.5);
  histetaMC_ = new TH1F("etaMC","MC particle eta",100,-2.5,2.5);
  histetaclu_ = new TH1F("clus eta","supercluster eta",100,-2.5,2.5);
  histeffeta_ = new TH1F("eta eff","seed effciency vs eta",100,-2.5,2.5);
  histq_ = new TH1F("q","seed charge",100,-2.5,2.5);
  histeoverp_ = new TH1F("E/p","seed E/p",100,0.,10.);
  histnbseeds_ = new TH1I("nrs","Nr of seeds ",50,0.,25.);
  histnbclus_ = new TH1I("nrclus","Nr of superclusters ",50,0.,25.);
  histnrseeds_ = new TH1I("ns","Nr of seeds if clusters",50,0.,25.);
}     

void
ElectronPixelSeedAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& iSetup)
{
  
  // rereads the seeds for test purposes
  typedef edm::OwnVector<TrackingRecHit> recHitContainer;
  typedef recHitContainer::const_iterator const_iterator;
  typedef std::pair<const_iterator,const_iterator> range;

  // get seeds
  
  edm::Handle<ElectronPixelSeedCollection> elSeeds;
  e.getByLabel(seedProducer_,seedLabel_,elSeeds); 
  edm::LogInfo("")<<"\n\n =================> Treating event "<<e.id()<<" Number of seeds "<<elSeeds.product()->size();
  int is=0;

  for( ElectronPixelSeedCollection::const_iterator MyS= (*elSeeds).begin(); MyS != (*elSeeds).end(); ++MyS) {
    
    LogDebug("") <<"\nSeed nr "<<is<<": ";
    range r=(*MyS).recHits();
     LogDebug("")<<" Number of RecHits= "<<(*MyS).nHits();
    const GeomDet *det=0;
    for (TrackingRecHitCollection::const_iterator rhits=r.first; rhits!=r.second; rhits++) {
      det = pDD->idToDet(((*rhits)).geographicalId());
       LogDebug("") <<" SiPixelHit   local x,y,z "<<(*rhits).localPosition()<<" det "<<(*rhits).geographicalId().det()<<" subdet "<<(*rhits).geographicalId().subdetId();
       LogDebug("") <<" SiPixelHit   global  "<<det->toGlobal((*rhits).localPosition());
    }   
    
    // state on last det
    TrajectoryStateOnSurface t= transformer_.transientState((*MyS).startingState(), &(det->surface()), &(*theMagField));

    // debug
    
     LogDebug("")<<" ElectronPixelSeed outermost state position: "<<t.globalPosition();
     LogDebug("")<<" ElectronPixelSeed outermost state momentum: "<<t.globalMomentum();
     edm::Ref<SuperClusterCollection> theClus=(*MyS).superCluster();
     LogDebug("")<<" ElectronPixelSeed superCluster energy: "<<theClus->energy()<<", position: "<<theClus->position();
     LogDebug("")<<" ElectronPixelSeed outermost state Pt: "<<t.globalMomentum().perp();
     LogDebug("")<<" ElectronPixelSeed supercluster Et: "<<theClus->energy()*sin(2.*atan(exp(-theClus->position().eta())));
     LogDebug("")<<" ElectronPixelSeed outermost momentum direction eta: "<<t.globalMomentum().eta();
     LogDebug("")<<" ElectronPixelSeed supercluster eta: "<<theClus->position().eta();
     LogDebug("")<<" ElectronPixelSeed seed charge: "<<(*MyS).getCharge();
     LogDebug("")<<" ElectronPixelSeed E/p: "<<theClus->energy()/t.globalMomentum().mag();

    // fill the tree and histos
    
    histpt_->Fill(t.globalMomentum().perp());
    histetclu_->Fill(theClus->energy()*sin(2.*atan(exp(-theClus->position().eta()))));
    histeta_->Fill(t.globalMomentum().eta());
    histetaclu_->Fill(theClus->position().eta());
    histq_->Fill((*MyS).getCharge());
    histeoverp_->Fill(theClus->energy()/t.globalMomentum().mag());   
    
    if (is<10) {
      superclusterEnergy[is] = theClus->energy();
      superclusterEta[is] = theClus->position().eta();
      superclusterPhi[is] = theClus->position().phi();
      superclusterEt[is] = theClus->energy()*sin(2.*atan(exp(-theClus->position().eta())));
      seedMomentum[is] = t.globalMomentum().mag();
      seedEta[is] = t.globalMomentum().eta();
      seedPhi[is] = t.globalMomentum().phi();
      seedPt[is] = t.globalMomentum().perp();
      seedQ[is] = (*MyS).getCharge();
    }
    
    is++;
    
  }
  
  histnbseeds_->Fill(elSeeds.product()->size());

  // get input clusters 

  edm::Handle<SuperClusterCollection> clusters;
  //CC to be changed according to supercluster input
  e.getByLabel("correctedHybridSuperClusterProducer", "correctedHybridSuperClusterCollection", clusters); 
  histnbclus_->Fill(clusters.product()->size());
  if (clusters.product()->size()>0) histnrseeds_->Fill(elSeeds.product()->size());
  
  // get MC information
  
  edm::Handle<edm::HepMCProduct> HepMCEvt;
  // this one is empty branch in current test files
  //e.getByLabel("VtxSmeared", "", HepMCEvt);
  e.getByLabel("source", "", HepMCEvt);
  
  const HepMC::GenEvent* MCEvt = HepMCEvt->GetEvent();
  int ip=0;
  for (HepMC::GenEvent::particle_const_iterator partIter = MCEvt->particles_begin();
   partIter != MCEvt->particles_end(); ++partIter) { 

    for (HepMC::GenEvent::vertex_const_iterator vertIter = MCEvt->vertices_begin();
     vertIter != MCEvt->vertices_end(); ++vertIter) {

      CLHEP::HepLorentzVector creation = (*partIter)->CreationVertex();
      CLHEP::HepLorentzVector momentum = (*partIter)->Momentum();
      HepPDT::ParticleID id = (*partIter)->particleID();  // electrons and positrons are 11 and -11
      LogDebug("")  << "MC particle id " << id.pid() << ", creationVertex " << creation << " cm, initialMomentum " << momentum << " GeV/c" << std::endl;
      if (id == 11 || id == -11) {
	histptMC_->Fill(momentum.perp());
	histetaMC_->Fill(momentum.pseudoRapidity());
	histeMC_->Fill(momentum.rho());
	if (ip<10) {
	  mcEnergy[ip] = momentum.rho();
	  mcEta[ip] = momentum.pseudoRapidity();
	  mcPhi[ip] = momentum.phi();
	  mcPt[ip] = momentum.perp();
	  mcQ[ip] = ((id == 11) ? -1.: +1.);
	}
      }

    }
    
    ip++;
   
  }  
    
  tree_->Fill();
  
}


