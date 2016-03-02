// -*- C++ -*-
//
// Package:    HiSignalGenJetProducer
// Class:      HiSignalGenJetProducer
// 
/**\class HiSignalGenJetProducer HiSignalGenJetProducer.cc yetkin/HiSignalGenJetProducer/src/HiSignalGenJetProducer.cc

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

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"

using namespace std;
using namespace edm;


//
// class decleration
//

class HiSignalGenJetProducer : public edm::EDProducer {
public:
      explicit HiSignalGenJetProducer(const edm::ParameterSet&);
      ~HiSignalGenJetProducer();

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      // ----------member data ---------------------------

   edm::EDGetTokenT<edm::View<reco::GenJet> > jetSrc_;

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

HiSignalGenJetProducer::HiSignalGenJetProducer(const edm::ParameterSet& iConfig) :
  jetSrc_(consumes<edm::View<reco::GenJet> >(iConfig.getParameter<edm::InputTag>("src")))
{
  std::string alias = (iConfig.getParameter<InputTag>( "src")).label();
  produces<reco::GenJetCollection>().setBranchAlias (alias);
}

HiSignalGenJetProducer::~HiSignalGenJetProducer()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to produce the data  ------------

void
HiSignalGenJetProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco;

   auto_ptr<GenJetCollection> jets;
   jets = auto_ptr<GenJetCollection>(new GenJetCollection);
   
   edm::Handle<edm::View<GenJet> > genjets;
   iEvent.getByToken(jetSrc_,genjets);

   int jetsize = genjets->size();

   vector<int> selection;
   for(int ijet = 0; ijet < jetsize; ++ijet){
      selection.push_back(-1);
   }

   vector<int> selectedIndices;
   vector<int> removedIndices;

   for(int ijet = 0; ijet < jetsize; ++ijet){

     const GenJet* jet1 = &((*genjets)[ijet]);

     const GenParticle* gencon = jet1->getGenConstituent(0);

     if(gencon == 0) throw cms::Exception("GenConstituent","GenJet is missing its constituents");
     else if(gencon->collisionId() == 0){
       jets->push_back(*jet1);
       selection[ijet] = 1;
     }else{
       selection[ijet] = 0;
       removedIndices.push_back(ijet);
     }
   }

   iEvent.put(jets);

}

DEFINE_FWK_MODULE(HiSignalGenJetProducer);


