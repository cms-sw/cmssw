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
// $Id: PATHemisphereProducer.cc,v 1.3 2008/04/08 09:02:18 trommers Exp $
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
  _seedMethod    ( iConfig.getParameter<int>("seedMethod") ),
  _combinationMethod ( iConfig.getParameter<int>("combinationMethod") )



  //  _EJselectionCfg(iConfig.getParameter<edm::ParameterSet>("ElectronJetCrossCleaning")),    
  // _ElectronJetCC(reco::modules::make<ElectronJetCrossCleaner>(_EJselectionCfg))
{
  //produces<std::vector<std::double
  ///produces cross-cleaned collections of above objects
  //Alternative: produce cross-cleaning decision & MET correction per object
//    produces<HemiAxis>("hemi1"); //hemisphere 1 axis
//    produces<HemiAxis>("hemi2"); //hemisphere 1 axis
 
//   produces<std::vector<pat::Jet> >();
//   produces<std::vector<pat::MET> >();
//   produces<std::vector<pat::Muon> >();
//   produces<std::vector<pat::Electron> >();
//   produces<std::vector<pat::Photon> >();
//   produces<std::vector<pat::Tau> >();

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
     vPx.push_back((*pJets)[i].px());
     vPy.push_back((*pJets)[i].py());
     vPz.push_back((*pJets)[i].pz());
     vE.push_back((*pJets)[i].energy());
     componentRefs_.push_back(pJets->refAt(i));
   }

   for(int i = 0; i < (int) (*pMuons).size() ; i++){
     vPx.push_back((*pMuons)[i].px());
     vPy.push_back((*pMuons)[i].py());
     vPz.push_back((*pMuons)[i].pz());
     vE.push_back((*pMuons)[i].energy());
     componentRefs_.push_back(pMuons->refAt(i));
   }
  
   for(int i = 0; i < (int) (*pElectrons).size() ; i++){
     vPx.push_back((*pElectrons)[i].px());
     vPy.push_back((*pElectrons)[i].py());
     vPz.push_back((*pElectrons)[i].pz());
     vE.push_back((*pElectrons)[i].energy());
     componentRefs_.push_back(pElectrons->refAt(i));
   } 

   for(int i = 0; i < (int) (*pPhotons).size() ; i++){
     vPx.push_back((*pPhotons)[i].px());
     vPy.push_back((*pPhotons)[i].py());
     vPz.push_back((*pPhotons)[i].pz());
     vE.push_back((*pPhotons)[i].energy());
     componentRefs_.push_back(pPhotons->refAt(i));
   } 

   //aren't taus included in jets?
   for(int i = 0; i < (int) (*pTaus).size() ; i++){
     vPx.push_back((*pTaus)[i].px());
     vPy.push_back((*pTaus)[i].py());
     vPz.push_back((*pTaus)[i].pz());
     vE.push_back((*pTaus)[i].energy());
     componentRefs_.push_back(pTaus->refAt(i));
   }  

   // create product
   std::auto_ptr< std::vector<Hemisphere> > hemispheres(new std::vector<Hemisphere>);;
   hemispheres->reserve(2);

  //calls HemiAlgorithm for seed method 3 (transv. inv. Mass) and association method 3 (Lund algo)
  HemisphereAlgo myHemi(vPx,vPy,vPz,vE,_seedMethod,_combinationMethod);

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
      (*hemispheres)[0].addDaughter(componentRefs_[i]);
    }
    else {
      (*hemispheres)[1].addDaughter(componentRefs_[i]);
    }
  }

//   std::auto_ptr<HemiAxis > hemiAxis1(new HemiAxis(vA1));
//   std::auto_ptr<HemiAxis > hemiAxis2(new HemiAxis(vA2));

 

  //  hemi1->push_back(vA1);
  // hemi2->push_back(vA2);

//    iEvent.put(hemiAxis1,"hemi1");
//    iEvent.put(hemiAxis2,"hemi2");
  iEvent.put(hemispheres);

  //clean up
//     delete myHemi;
    vPx.clear();
    vPy.clear();
    vPz.clear();
    vE.clear();
    vgroups.clear();
    componentRefs_.clear();
}



// ------------ method called once each job just before starting event loop  ------------
void 
PATHemisphereProducer::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
PATHemisphereProducer::endJob() {
  
}

//define this as a plug-in
DEFINE_FWK_MODULE(PATHemisphereProducer);
