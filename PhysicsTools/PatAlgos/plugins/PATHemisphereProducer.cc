
// -*- C++ -*-
//
// Package:    PatShapeAna
// Class:      PatShapeAna
// 
/**\class PatShapeAna PatShapeAna.cc PhysicsTools/PatShapeAna/src/PatShapeAna.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tanja Rommerskirchen
//         Created:  Sat Mar 22 12:58:04 CET 2008
// $Id: PATHemisphereProducer.cc,v 1.9 2010/01/11 13:36:48 hegner Exp $
//
//


//system
#include <vector>
#include <memory>
//PAT
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Hemisphere.h"
//DataFormats
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Candidate/interface/Particle.h"
//User
#include  "PhysicsTools/PatAlgos/plugins/PATHemisphereProducer.h"


using namespace pat;


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PATHemisphereProducer::PATHemisphereProducer(const edm::ParameterSet& iConfig) :
  _patJets       ( iConfig.getParameter<edm::InputTag>( "patJets" ) ),
  _patMuons      ( iConfig.getParameter<edm::InputTag>( "patMuons" ) ),
  _patElectrons  ( iConfig.getParameter<edm::InputTag>( "patElectrons" ) ),
  _patPhotons    ( iConfig.getParameter<edm::InputTag>( "patPhotons" ) ),
  _patTaus       ( iConfig.getParameter<edm::InputTag>( "patTaus" ) ),

  _minJetEt       ( iConfig.getParameter<double>("minJetEt") ),
  _minMuonEt       ( iConfig.getParameter<double>("minMuonEt") ),
  _minElectronEt       ( iConfig.getParameter<double>("minElectronEt") ),
  _minTauEt       ( iConfig.getParameter<double>("minTauEt") ), 
  _minPhotonEt       ( iConfig.getParameter<double>("minPhotonEt") ),

  _maxJetEta       ( iConfig.getParameter<double>("maxJetEta") ),
  _maxMuonEta       ( iConfig.getParameter<double>("maxMuonEta") ),
  _maxElectronEta       ( iConfig.getParameter<double>("maxElectronEta") ),
  _maxTauEta       ( iConfig.getParameter<double>("maxTauEta") ), 
  _maxPhotonEta       ( iConfig.getParameter<double>("maxPhotonEta") ),

  _seedMethod    ( iConfig.getParameter<int>("seedMethod") ),
  _combinationMethod ( iConfig.getParameter<int>("combinationMethod") )

{


  produces< std::vector<pat::Hemisphere> >();
}


PATHemisphereProducer::~PATHemisphereProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
PATHemisphereProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;

   //Jets   
   Handle<reco::CandidateView> pJets;
   iEvent.getByLabel(_patJets,pJets);

   //Muons   
   Handle<reco::CandidateView> pMuons;
   iEvent.getByLabel(_patMuons,pMuons);

   //Electrons   
   Handle<reco::CandidateView> pElectrons;
   iEvent.getByLabel(_patElectrons,pElectrons);

   //Photons   
   Handle<reco::CandidateView> pPhotons;
   iEvent.getByLabel(_patPhotons,pPhotons);

   //Taus   
   Handle<reco::CandidateView> pTaus;
   iEvent.getByLabel(_patTaus,pTaus);


   //fill e,p vector with information from all objects (hopefully cleaned before)
   for(int i = 0; i < (int) (*pJets).size() ; i++){
     if((*pJets)[i].pt() <  _minJetEt || fabs((*pJets)[i].eta()) >  _maxJetEta) continue;
   
     componentPtrs_.push_back(pJets->ptrAt(i));
   }

   for(int i = 0; i < (int) (*pMuons).size() ; i++){
     if((*pMuons)[i].pt() <  _minMuonEt || fabs((*pMuons)[i].eta()) >  _maxMuonEta) continue; 
 
     componentPtrs_.push_back(pMuons->ptrAt(i));
   }
  
   for(int i = 0; i < (int) (*pElectrons).size() ; i++){
     if((*pElectrons)[i].pt() <  _minElectronEt || fabs((*pElectrons)[i].eta()) >  _maxElectronEta) continue;  
    
     componentPtrs_.push_back(pElectrons->ptrAt(i));
   } 

   for(int i = 0; i < (int) (*pPhotons).size() ; i++){
     if((*pPhotons)[i].pt() <  _minPhotonEt || fabs((*pPhotons)[i].eta()) >  _maxPhotonEta) continue;   
    
     componentPtrs_.push_back(pPhotons->ptrAt(i));
   } 

   //aren't taus included in jets?
   for(int i = 0; i < (int) (*pTaus).size() ; i++){
     if((*pTaus)[i].pt() <  _minTauEt || fabs((*pTaus)[i].eta()) >  _maxTauEta) continue;   
    
     componentPtrs_.push_back(pTaus->ptrAt(i));
   }  

   // create product
   std::auto_ptr< std::vector<Hemisphere> > hemispheres(new std::vector<Hemisphere>);;
   hemispheres->reserve(2);

  //calls HemiAlgorithm for seed method 3 (transv. inv. Mass) and association method 3 (Lund algo)
  HemisphereAlgo myHemi(componentPtrs_,_seedMethod,_combinationMethod);

  //get Hemisphere Axis 
  vA1 = myHemi.getAxis1();
  vA2 = myHemi.getAxis2();

  reco::Particle::LorentzVector p1(vA1[0]*vA1[3],vA1[1]*vA1[3],vA1[2]*vA1[3],vA1[4]);
  hemispheres->push_back(Hemisphere(p1));

  reco::Particle::LorentzVector p2(vA2[0]*vA2[3],vA2[1]*vA2[3],vA2[2]*vA2[3],vA2[4]);
  hemispheres->push_back(Hemisphere(p2));
 
  //get information to which Hemisphere each object belongs
  vgroups = myHemi.getGrouping(); 

  for ( unsigned int i=0; i<vgroups.size(); ++i ) {
    if ( vgroups[i]==1 ) {
      (*hemispheres)[0].addDaughter(componentPtrs_[i]);
    }
    else {
      (*hemispheres)[1].addDaughter(componentPtrs_[i]);
    }
  }


  iEvent.put(hemispheres);

  //clean up

    vPx.clear();
    vPy.clear();
    vPz.clear();
    vE.clear();
    vgroups.clear();
    componentPtrs_.clear();
}



// ------------ method called once each job just after ending the event loop  ------------
void 
PATHemisphereProducer::endJob() {
  
}

//define this as a plug-in
DEFINE_FWK_MODULE(PATHemisphereProducer);
