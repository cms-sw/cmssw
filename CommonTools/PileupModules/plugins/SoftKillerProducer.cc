// -*- C++ -*-
//
// Package:    CommonTools/PileupModules
// Class:      SoftKillerProducer
// 
/**\class SoftKillerProducer SoftKillerProducer.cc CommonTools/PileupModules/plugins/SoftKillerProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Salvatore Rappoccio
//         Created:  Wed, 22 Oct 2014 15:14:20 GMT
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
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "fastjet/contrib/SoftKiller.hh"


//
// class declaration
//

class SoftKillerProducer : public edm::EDProducer {
public:

  typedef std::vector< edm::FwdPtr<reco::PFCandidate> >  PFCollection;
  typedef edm::View<reco::PFCandidate>                   PFView;

  explicit SoftKillerProducer(const edm::ParameterSet&);
  ~SoftKillerProducer();

private:
  virtual void beginJob() override;
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;

  edm::EDGetTokenT< PFCollection > tokenPFCandidates_;


  double Rho_EtaMax_;
  double rParam_;
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
SoftKillerProducer::SoftKillerProducer(const edm::ParameterSet& iConfig) : 
  Rho_EtaMax_( iConfig.getParameter<double>("Rho_EtaMax") ),
  rParam_ ( iConfig.getParameter<double>("rParam") )
{

  tokenPFCandidates_
    = consumes<PFCollection>(iConfig.getParameter<edm::InputTag>("PFCandidates"));

  produces< PFCollection >();

}


SoftKillerProducer::~SoftKillerProducer()
{

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
SoftKillerProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  std::auto_ptr< PFCollection > pOutput( new PFCollection );

  // get PF Candidates
  edm::Handle<PFCollection> pfCandidates;
  iEvent.getByToken( tokenPFCandidates_, pfCandidates);

  std::vector<fastjet::PseudoJet> fjInputs;
  for ( auto i = pfCandidates->begin(), 
	  ibegin = pfCandidates->begin(),
	  iend = pfCandidates->end(); i != iend; ++i ) {
    fjInputs.push_back( fastjet::PseudoJet( (*i)->px(), (*i)->py(), (*i)->pz(), (*i)->energy() ) );
    fjInputs.back().set_user_index( i - ibegin );
  }

  // soft killer:
  fastjet::contrib::SoftKiller soft_killer(Rho_EtaMax_, rParam_);

  double pt_threshold = 0.;
  std::vector<fastjet::PseudoJet> soft_killed_event;
  soft_killer.apply(fjInputs, soft_killed_event, pt_threshold);

  for ( auto j = soft_killed_event.begin(),
	  jend = soft_killed_event.end(); j != jend; ++j ) {
    pOutput->push_back( (*pfCandidates)[ j->user_index() ] );
  }

  iEvent.put( pOutput );

}

// ------------ method called once each job just before starting event loop  ------------
void 
SoftKillerProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SoftKillerProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(SoftKillerProducer);
