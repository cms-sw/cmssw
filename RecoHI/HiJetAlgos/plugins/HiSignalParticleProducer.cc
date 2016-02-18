// -*- C++ -*-
//
// Package:    HiSignalParticleProducer
// Class:      HiSignalParticleProducer
// 
/**\class HiSignalParticleProducer HiSignalParticleProducer.cc yetkin/HiSignalParticleProducer/src/HiSignalParticleProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Yetkin Yilmaz
//         Created:  Tue Jul 21 04:26:01 EDT 2009
//
//

// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"

using namespace std;
using namespace edm;


//
// class decleration
//

class HiSignalParticleProducer : public edm::EDProducer {
public:
      explicit HiSignalParticleProducer(const edm::ParameterSet&);
      ~HiSignalParticleProducer();

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      // ----------member data ---------------------------

   edm::EDGetTokenT<edm::View<reco::GenParticle> > jetSrc_;
   double deltaR_;
   double ptCut_;
   bool makeNew_;
   bool fillDummy_;

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

HiSignalParticleProducer::HiSignalParticleProducer(const edm::ParameterSet& iConfig) :
  jetSrc_(consumes<edm::View<reco::GenParticle> >(iConfig.getParameter<edm::InputTag>("src"))),
  deltaR_(iConfig.getParameter<double>("deltaR")),
  ptCut_(iConfig.getParameter<double>("ptCut")),
  makeNew_(iConfig.getUntrackedParameter<bool>("createNewCollection",true)),
  fillDummy_(iConfig.getUntrackedParameter<bool>("fillDummyEntries",true))
{
  std::string alias = (iConfig.getParameter<InputTag>( "src")).label();
  produces<reco::GenParticleCollection>().setBranchAlias (alias);
}

HiSignalParticleProducer::~HiSignalParticleProducer()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to produce the data  ------------

void
HiSignalParticleProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco;

   auto_ptr<GenParticleCollection> jets;
   jets = auto_ptr<GenParticleCollection>(new GenParticleCollection);
   
   edm::Handle<edm::View<GenParticle> > genjets;
   iEvent.getByToken(jetSrc_,genjets);

   int jetsize = genjets->size();

   vector<int> selection;
   for(int ijet = 0; ijet < jetsize; ++ijet){
      selection.push_back(-1);
   }

   vector<int> selectedIndices;
   vector<int> removedIndices;

   for(int ijet = 0; ijet < jetsize; ++ijet){

     const GenParticle* jet1 = &((*genjets)[ijet]);
     if(jet1->collisionId() == 0){
       jets->push_back(*jet1);
       selection[ijet] = 1;
     }else{
       selection[ijet] = 0;
       removedIndices.push_back(ijet);
     }
   }

   iEvent.put(jets);

}

DEFINE_FWK_MODULE(HiSignalParticleProducer);


