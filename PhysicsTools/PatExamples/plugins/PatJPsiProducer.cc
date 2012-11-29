// -*- C++ -*-
//
// Package:    PatJPsiProducer
// Class:      PatJPsiProducer
// 
/**\class PatJPsiProducer PatJPsiProducer.cc Analysis/PatJPsiProducer/src/PatJPsiProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  "Salvatore Rappoccio"
//         Created:  Mon Sep 28 12:53:57 CDT 2009
// $Id: PatJPsiProducer.cc,v 1.2 2009/10/20 12:24:28 hegner Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include "DataFormats/Math/interface/deltaR.h"


#include <TLorentzVector.h>


#include <vector>

//
// class decleration
//

class PatJPsiProducer : public edm::EDProducer {
   public:
      explicit PatJPsiProducer(const edm::ParameterSet&);
      ~PatJPsiProducer();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------

  edm::InputTag    muonSrc_;
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
PatJPsiProducer::PatJPsiProducer(const edm::ParameterSet& iConfig) :
  muonSrc_ ( iConfig.getParameter<edm::InputTag>("muonSrc") )
{
  produces<std::vector<pat::CompositeCandidate> > ();
  
}


PatJPsiProducer::~PatJPsiProducer()
{
 
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
PatJPsiProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  std::auto_ptr<std::vector<pat::CompositeCandidate> > jpsiCands( new std::vector<pat::CompositeCandidate> );
  edm::Handle<edm::View<pat::Muon> > h_muons;
  iEvent.getByLabel( muonSrc_, h_muons );
  std::cout<<"valid?"<< h_muons.isValid()<<" size?"<<  h_muons->size();
  if ( h_muons.isValid() && h_muons->size() > 1 ) {

      for ( edm::View<pat::Muon>::const_iterator muonsBegin = h_muons->begin(),
		muonsEnd = h_muons->end(), imuon = muonsBegin;
	    imuon != muonsEnd - 1; ++imuon ) {
	  
	  for ( edm::View<pat::Muon>::const_iterator jmuon = imuon + 1;
		jmuon != muonsEnd; ++jmuon ) {
	      if ( imuon->charge() * jmuon->charge() < 0 ) {
		  
		  //A composite Candidate is very useful to build event hypothesis and cut on combined object information.
		  pat::CompositeCandidate jpsi;
		  jpsi.addDaughter( *imuon, "mu1");
		  jpsi.addDaughter( *jmuon, "mu2");
		  
		  AddFourMomenta addp4;
		  addp4.set( jpsi );
		  
		  double dR = reco::deltaR<pat::Muon,pat::Muon>( *imuon, *jmuon );
		  
		  // Analogue to any other PAT object we can add our own information into the object via addUserFloat/Int/Data()
		  jpsi.addUserFloat("dR", dR );
		  
		  jpsiCands->push_back( jpsi );
		  
	      }
	  }
      }
  }
  iEvent.put( jpsiCands );
 
}

// ------------ method called once each job just before starting event loop  ------------
void 
PatJPsiProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
PatJPsiProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(PatJPsiProducer);
