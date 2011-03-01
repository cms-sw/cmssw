// -*- C++ -*-
//
// Package:    TestDummyPFCandidateAnalyzer
// Class:      TestDummyPFCandidateAnalyzer
// 
/**\class TestDummyPFCandidateAnalyzer TestDummyPFCandidateAnalyzer.cc DataFormats/TestDummyPFCandidateAnalyzer/src/TestDummyPFCandidateAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Christopher Jones,40 5-A15,+41227674799,
//         Created:  Tue Mar  1 19:30:21 CET 2011
// $Id$
//
//


// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
//
// class declaration
//

class TestDummyPFCandidateAnalyzer : public edm::EDAnalyzer {
   public:
      explicit TestDummyPFCandidateAnalyzer(const edm::ParameterSet&);
      ~TestDummyPFCandidateAnalyzer();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
      edm::InputTag m_tag;
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
TestDummyPFCandidateAnalyzer::TestDummyPFCandidateAnalyzer(const edm::ParameterSet& iConfig)
: m_tag(iConfig.getUntrackedParameter<edm::InputTag>("tag"))
{
   //now do what ever initialization is needed

}


TestDummyPFCandidateAnalyzer::~TestDummyPFCandidateAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//
#define TEST_VALID( _test_ ) if( not (_test_) ) { throw cms::Exception("TestFailed") << # _test_ ;}

// ------------ method called to for each event  ------------
void
TestDummyPFCandidateAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;



   Handle<std::vector<reco::Track> > pTracks;
   iEvent.getByLabel(m_tag,pTracks);

   Handle<std::vector<reco::Muon> > pMuons;
   iEvent.getByLabel(m_tag,pMuons);

   Handle<std::vector<reco::PFCandidate> > pCands;
   iEvent.getByLabel(m_tag,pCands);
   
   TEST_VALID(pCands->size() == 2);
   TEST_VALID(pMuons->size() == 2);
   TEST_VALID(pTracks->size() == 4);
   
   TEST_VALID(pCands->at(0).trackRef().get() == &(pTracks->at(2)));
   TEST_VALID(pCands->at(0).muonRef().isNull());
   TEST_VALID(pCands->at(1).muonRef().get() == &(pMuons->at(1)));
   TEST_VALID(pCands->at(1).trackRef().isNull());
}


// ------------ method called once each job just before starting event loop  ------------
void 
TestDummyPFCandidateAnalyzer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TestDummyPFCandidateAnalyzer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestDummyPFCandidateAnalyzer);
