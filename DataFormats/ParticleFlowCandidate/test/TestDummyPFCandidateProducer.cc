// -*- C++ -*-
//
// Package:    TestDummyPFCandidateProducer
// Class:      TestDummyPFCandidateProducer
// 
/**\class TestDummyPFCandidateProducer TestDummyPFCandidateProducer.cc DataFormats/TestDummyPFCandidateProducer/src/TestDummyPFCandidateProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Christopher Jones,40 5-A15,+41227674799,
//         Created:  Tue Mar  1 18:16:40 CET 2011
// $Id$
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

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
//
// class declaration
//

class TestDummyPFCandidateProducer : public edm::EDProducer {
   public:
      explicit TestDummyPFCandidateProducer(const edm::ParameterSet&);
      ~TestDummyPFCandidateProducer();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
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
TestDummyPFCandidateProducer::TestDummyPFCandidateProducer(const edm::ParameterSet& iConfig)
{
   //register your products
  produces<std::vector<reco::PFCandidate> >();
  produces<std::vector<reco::Track> >();
  produces<std::vector<reco::Muon> >();
   //now do what ever other initialization is needed
  
}


TestDummyPFCandidateProducer::~TestDummyPFCandidateProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
TestDummyPFCandidateProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   
   std::auto_ptr<std::vector<reco::Track> > pTracks ( new std::vector<reco::Track> );
   pTracks->resize(4);
   std::auto_ptr<std::vector<reco::Muon> > pMuons ( new std::vector<reco::Muon> );
   pMuons->resize(2);
   
   edm::OrphanHandle<std::vector<reco::Track> > hTracks( iEvent.put(pTracks));
   edm::OrphanHandle<std::vector<reco::Muon> > hMuons( iEvent.put(pMuons));
   
   
   std::auto_ptr<std::vector<reco::PFCandidate> > pCands( new std::vector<reco::PFCandidate> );
   
   pCands->push_back( reco::PFCandidate(-1., reco::PFCandidate::LorentzVector(1.,1.,1.,2.), reco::PFCandidate::e));
   pCands->back().setTrackRef( reco::TrackRef(hTracks,2));
   
   pCands->push_back( reco::PFCandidate(1., reco::PFCandidate::LorentzVector(1.,1.,1.,2.), reco::PFCandidate::mu));
   pCands->back().setMuonRef( reco::MuonRef(hMuons,1));

   iEvent.put(pCands);
 
}

// ------------ method called once each job just before starting event loop  ------------
void 
TestDummyPFCandidateProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TestDummyPFCandidateProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestDummyPFCandidateProducer);
