// -*- C++ -*-
//
// Package:    RecHitComparison
// Class:      RecHitComparison
//
/**\class RecHitComparison RecHitComparison.cc CmsHi/RecHitComparison/src/RecHitComparison.cc

   Description: [one line class summary]

   Implementation:
   [Notes on implementation]
*/
//
// Original Author:  Yetkin Yilmaz
//         Created:  Tue Sep  7 11:38:19 EDT 2010
// $Id: RecHitComparison.cc,v 1.10 2011/03/30 21:21:10 yilmaz Exp $
//
//


// system include files
#include <memory>
#include <vector>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FWLite/interface/ChainEvent.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

#include "TNtuple.h"

using namespace std;

//
// class declaration
//

class RecHitComparison : public edm::EDAnalyzer {
public:
  explicit RecHitComparison(const edm::ParameterSet&);
  ~RecHitComparison();


private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  // ----------member data ---------------------------
  edm::Handle<vector<double> > ktRhos;
  edm::Handle<vector<double> > akRhos;

  edm::Handle<edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> > > ebHits1;
  edm::Handle<edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> > > ebHits2;
  edm::Handle<edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> > > eeHits1;
  edm::Handle<edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> > > eeHits2;

  edm::Handle<HFRecHitCollection> hfHits1;
  edm::Handle<HFRecHitCollection> hfHits2;
  edm::Handle<HBHERecHitCollection> hbheHits1;
  edm::Handle<HBHERecHitCollection> hbheHits2;

  edm::Handle<reco::BasicClusterCollection> bClusters1;
  edm::Handle<reco::BasicClusterCollection> bClusters2;


  typedef vector<EcalRecHit>::const_iterator EcalIterator;
  typedef vector<HFRecHit>::const_iterator HFIterator;
  typedef vector<HBHERecHit>::const_iterator HBHEIterator;

  edm::Handle<reco::CaloJetCollection> signalJets;

  edm::InputTag HcalRecHitHFSrc1_;
  edm::InputTag HcalRecHitHFSrc2_;
  edm::InputTag HcalRecHitHBHESrc1_;
  edm::InputTag HcalRecHitHBHESrc2_;
  edm::InputTag EBSrc1_;
  edm::InputTag EBSrc2_;
  edm::InputTag EESrc1_;
  edm::InputTag EESrc2_;

  edm::InputTag BCSrc1_;
  edm::InputTag BCSrc2_;

  edm::InputTag signalTag_;

  TNtuple* ntBC;
  TNtuple* ntEB;
  TNtuple* ntEE;
  TNtuple* ntHBHE;
  TNtuple* ntHF;
  TNtuple* ntjet;

  double cone;
  bool jetsOnly_;
  bool doBasicClusters_;
  bool doJetCone_;

  edm::Service<TFileService> fs;
  const CaloGeometry *geo;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
RecHitComparison::RecHitComparison(const edm::ParameterSet& iConfig) :
  cone(0.5),
  geo(0)
{
  //now do what ever initialization is needed
  jetsOnly_ = iConfig.getUntrackedParameter<bool>("jetsOnly",false);
  doBasicClusters_ = iConfig.getUntrackedParameter<bool>("doBasicClusters",false);

  doJetCone_ = iConfig.getUntrackedParameter<bool>("doJetCone",false);
  signalTag_ = iConfig.getUntrackedParameter<edm::InputTag>("signalJets",edm::InputTag("iterativeCone5CaloJets","","SIGNAL"));

  if(!doJetCone_) jetsOnly_ = 0;

  HcalRecHitHFSrc1_ = iConfig.getUntrackedParameter<edm::InputTag>("hcalHFRecHitSrc1",edm::InputTag("hfreco"));
  HcalRecHitHFSrc2_ = iConfig.getUntrackedParameter<edm::InputTag>("hcalHFRecHitSrc2",edm::InputTag("hfreco"));
  HcalRecHitHBHESrc1_ = iConfig.getUntrackedParameter<edm::InputTag>("hcalHBHERecHitSrc1",edm::InputTag("hbhereco"));
  HcalRecHitHBHESrc2_ = iConfig.getUntrackedParameter<edm::InputTag>("hcalHBHERecHitSrc2",edm::InputTag("hbhereco"));
  EBSrc1_ = iConfig.getUntrackedParameter<edm::InputTag>("EBRecHitSrc1",edm::InputTag("ecalRecHit","EcalRecHitsEB","RECOBKG"));
  EBSrc2_ = iConfig.getUntrackedParameter<edm::InputTag>("EBRecHitSrc2",edm::InputTag("ecalRecHit","EcalRecHitsEB","S"));
  EESrc1_ = iConfig.getUntrackedParameter<edm::InputTag>("EERecHitSrc1",edm::InputTag("ecalRecHit","EcalRecHitsEE","RECO"));
  EESrc2_ = iConfig.getUntrackedParameter<edm::InputTag>("EERecHitSrc2",edm::InputTag("ecalRecHit","EcalRecHitsEE","SIGNALONLY"));
  BCSrc1_ = iConfig.getUntrackedParameter<edm::InputTag>("BasicClusterSrc1",edm::InputTag("ecalRecHit","EcalRecHitsEB","RECO"));
  BCSrc2_ = iConfig.getUntrackedParameter<edm::InputTag>("BasicClusterSrc2",edm::InputTag("ecalRecHit","EcalRecHitsEB","SIGNALONLY"));

}


RecHitComparison::~RecHitComparison()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
RecHitComparison::analyze(const edm::Event& ev, const edm::EventSetup& iSetup)
{
  if(!geo){
    edm::ESHandle<CaloGeometry> pGeo;
    iSetup.get<CaloGeometryRecord>().get(pGeo);
    geo = pGeo.product();
  }

  using namespace edm;
  ev.getByLabel(EBSrc1_,ebHits1);
  ev.getByLabel(EBSrc2_,ebHits2);

  if(doJetCone_) ev.getByLabel(signalTag_,signalJets);

  ev.getByLabel(HcalRecHitHFSrc1_,hfHits1);
  ev.getByLabel(HcalRecHitHFSrc2_,hfHits2);
  ev.getByLabel(HcalRecHitHBHESrc1_,hbheHits1);
  ev.getByLabel(HcalRecHitHBHESrc2_,hbheHits2);
  ev.getByLabel(EESrc1_,eeHits1);
  ev.getByLabel(EESrc2_,eeHits2);

  if(doBasicClusters_){
    ev.getByLabel(BCSrc1_,bClusters1);
    ev.getByLabel(BCSrc2_,bClusters2);
  }

  vector<double> fFull;
  vector<double> f05;
  vector<double> f1;
  vector<double> f15;
  vector<double> f2;
  vector<double> f25;
  vector<double> f3;

  int njets = 0;

  if(doJetCone_) njets = signalJets->size();
  fFull.reserve(njets);
  f05.reserve(njets);
  f1.reserve(njets);
  f15.reserve(njets);
  f2.reserve(njets);
  f25.reserve(njets);
  f3.reserve(njets);

  if(doJetCone_){
    for(unsigned int j1 = 0 ; j1 < signalJets->size(); ++j1){
      fFull.push_back(0);
      f05.push_back(0);
      f1.push_back(0);
      f15.push_back(0);
      f2.push_back(0);
      f25.push_back(0);
      f3.push_back(0);
    }
  }

  for(unsigned int j1 = 0 ; j1 < ebHits1->size(); ++j1){

    const EcalRecHit& jet1 = (*ebHits1)[j1];
    double e1 = jet1.energy();

    const GlobalPoint& pos1=geo->getPosition(jet1.id());
    double eta1 = pos1.eta();
    double phi1 = pos1.phi();
    double et1 = e1*sin(pos1.theta());

    double drjet = -1;
    double jetpt = -1;
    bool isjet = false;
    //int matchedJet = -1;

    if(doJetCone_){
      for(unsigned int j = 0 ; j < signalJets->size(); ++j){
	const reco::CaloJet & jet = (*signalJets)[j];
	double dr = reco::deltaR(eta1,phi1,jet.eta(),jet.phi());
	if(dr < cone){
	  jetpt = jet.pt();
	  drjet = dr;
	  isjet = true;
	  //matchedJet = j;
	  fFull[j] += et1;

	  if(et1 > 0.5){
	    f05[j] += et1;
	  }
	  if(et1 > 1.){
	    f1[j] += et1;
	  }
	  if(et1 > 1.5){
	    f15[j] += et1;
	  }
	  if(et1 > 2){
	    f2[j] += et1;
	  }
	  if(et1 > 2.5){
	    f25[j] += et1;
	  }
	  if(et1 > 3.){
	    f3[j] += et1;
	  }
	}
      }
    }

    GlobalPoint pos2;
    double e2 = -1;
    EcalIterator hitit = ebHits2->find(jet1.id());
    if(hitit != ebHits2->end()){
      e2 = hitit->energy();
      pos2=geo->getPosition(hitit->id());
    }else{
      e2 = 0;
      pos2 = pos1;
    }

    double eta2 = pos2.eta();
    double phi2 = pos2.eta();
    double et2 = e2*sin(pos2.theta());
    if(!jetsOnly_ ||  isjet) ntEB->Fill(e1,et1,e2,et2,eta2,phi2,jetpt,drjet);
  }

  for(unsigned int i = 0; i < eeHits1->size(); ++i){
    const EcalRecHit & jet1= (*eeHits1)[i];
    double e1 = jet1.energy();
    const GlobalPoint& pos1=geo->getPosition(jet1.id());
    double eta1 = pos1.eta();
    double phi1 = pos1.phi();
    double et1 = e1*sin(pos1.theta());
    double drjet = -1;
    double jetpt = -1;
    bool isjet = false;
    //int matchedJet = -1;
    if(doJetCone_){
      for(unsigned int j = 0 ; j < signalJets->size(); ++j){
	const reco::CaloJet & jet = (*signalJets)[j];
	double dr = reco::deltaR(eta1,phi1,jet.eta(),jet.phi());
	if(dr < cone){
	  jetpt = jet.pt();
	  drjet = dr;
	  isjet = true;
	  //matchedJet = j;
	}
      }
    }

    GlobalPoint pos2;
    double e2 = -1;
    EcalIterator hitit = eeHits2->find(jet1.id());
    if(hitit != eeHits2->end()){
      e2 = hitit->energy();
      pos2=geo->getPosition(hitit->id());
    }else{
      e2 = 0;
      pos2 = pos1;
    }
    double eta2 = pos2.eta();
    double phi2 = pos2.eta();
    double et2 = e2*sin(pos2.theta());
    if(!jetsOnly_ || isjet) ntEE->Fill(e1,et1,e2,et2,eta2,phi2,jetpt,drjet);
  }

  for(unsigned int i = 0; i < hbheHits1->size(); ++i){
    const HBHERecHit & jet1= (*hbheHits1)[i];
    double e1 = jet1.energy();
    const GlobalPoint& pos1=geo->getPosition(jet1.id());
    double eta1 = pos1.eta();
    double phi1 = pos1.phi();
    double et1 = e1*sin(pos1.theta());
    double drjet = -1;
    double jetpt = -1;
    bool isjet = false;
    //int matchedJet = -1;
    if(doJetCone_){
      for(unsigned int j = 0 ; j < signalJets->size(); ++j){
	const reco::CaloJet & jet = (*signalJets)[j];
	double dr = reco::deltaR(eta1,phi1,jet.eta(),jet.phi());
	if(dr < cone){
	  jetpt = jet.pt();
	  drjet = dr;
	  isjet = true;
	  //matchedJet = j;
	}
      }
    }

    GlobalPoint pos2;
    double e2 = -1;
    HBHEIterator hitit = hbheHits2->find(jet1.id());
    if(hitit != hbheHits2->end()){
      e2 = hitit->energy();
      pos2=geo->getPosition(hitit->id());
    }else{
      e2 = 0;
      pos2 = pos1;
    }
    double eta2 = pos2.eta();
    double phi2 = pos2.eta();
    double et2 = e2*sin(pos2.theta());
    if(!jetsOnly_ || isjet) ntHBHE->Fill(e1,et1,e2,et2,eta2,phi2,jetpt,drjet);
  }

  for(unsigned int i = 0; i < hfHits1->size(); ++i){
    const HFRecHit & jet1= (*hfHits1)[i];
    double e1 = jet1.energy();
    const GlobalPoint& pos1=geo->getPosition(jet1.id());
    double eta1 = pos1.eta();
    double phi1 = pos1.phi();
    double et1 = e1*sin(pos1.theta());
    double drjet = -1;
    double jetpt = -1;
    bool isjet = false;
    //int matchedJet = -1;
    if(doJetCone_){
      for(unsigned int j = 0 ; j < signalJets->size(); ++j){
	const reco::CaloJet & jet = (*signalJets)[j];
	double dr = reco::deltaR(eta1,phi1,jet.eta(),jet.phi());
	if(dr < cone){
	  jetpt = jet.pt();
	  drjet = dr;
	  isjet = true;
	  //matchedJet = j;
	}
      }
    }
    GlobalPoint pos2;
    double e2 = -1;
    HFIterator hitit = hfHits2->find(jet1.id());
    if(hitit != hfHits2->end()){
      e2 = hitit->energy();
      pos2=geo->getPosition(hitit->id());
    }else{
      e2 = 0;
      pos2 = pos1;
    }
    double eta2 = pos2.eta();
    double phi2 = pos2.eta();
    double et2 = e2*sin(pos2.theta());
    if(!jetsOnly_ || isjet) ntHF->Fill(e1,et1,e2,et2,eta2,phi2,jetpt,drjet);
  }

  if(doJetCone_){
    for(unsigned int j1 = 0 ; j1 < signalJets->size(); ++j1){
      const reco::CaloJet & jet = (*signalJets)[j1];
      double em = (jet.emEnergyInEB() + jet.emEnergyInEE()) * sin(jet.theta());
      double emf = jet.emEnergyFraction();
      double pt = jet.pt();
      double eta = jet.eta();
      ntjet->Fill(pt,eta,fFull[j1],f05[j1],f1[j1],f15[j1],f2[j1],f25[j1],f3[j1],em,emf);
    }
  }

}


// ------------ method called once each job just before starting event loop  ------------
void
RecHitComparison::beginJob()
{
  ntEB = fs->make<TNtuple>("ntEB","","e1:et1:e2:et2:eta:phi:ptjet:drjet");
  ntEE = fs->make<TNtuple>("ntEE","","e1:et1:e2:et2:eta:phi:ptjet:drjet");
  ntHBHE = fs->make<TNtuple>("ntHBHE","","e1:et1:e2:et2:eta:phi:ptjet:drjet");
  ntHF = fs->make<TNtuple>("ntHF","","e1:et1:e2:et2:eta:phi:ptjet:drjet");

  ntBC = fs->make<TNtuple>("ntBC","","e1:et1:e2:et2:eta:phi:ptjet:drjet");

  ntjet = fs->make<TNtuple>("ntjet","","pt:eta:ethit:f05:f1:f15:f2:f25:f3:em:emf");

}

// ------------ method called once each job just after ending the event loop  ------------
void
RecHitComparison::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(RecHitComparison);
