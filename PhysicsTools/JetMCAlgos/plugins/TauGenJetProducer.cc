#include "PhysicsTools/JetMCAlgos/plugins/TauGenJetProducer.h"


#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"

#include "PhysicsTools/HepMCCandAlgos/interface/GenParticlesHelper.h"

using namespace std;
using namespace edm;
using namespace reco;

TauGenJetProducer::TauGenJetProducer(const edm::ParameterSet& iConfig) {
  

  inputTagGenParticles_ 
    = iConfig.getParameter<InputTag>("GenParticles");

  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);



  produces<GenJetCollection>();
}



TauGenJetProducer::~TauGenJetProducer() { }



void TauGenJetProducer::beginJob(const edm::EventSetup & es) { }


void TauGenJetProducer::produce(Event& iEvent, 
				const EventSetup& iSetup) {
  
  Handle<GenParticleCollection> genParticles;

  bool found = iEvent.getByLabel( inputTagGenParticles_, genParticles);
  
  if(!found ) {
    std::ostringstream  err;
    err<<" cannot get collection: "
       <<inputTagGenParticles_<<std::endl;
    edm::LogError("TauGenJetProducer")<<err.str();
    throw cms::Exception( "MissingProduct", err.str());
  }

  using namespace GenParticlesHelper;

  GenParticleRefVector allStatus2Taus;  
  findParticles( *genParticles,
		 allStatus2Taus, 15, 2);

  for( IGR iTau=allStatus2Taus.begin(); iTau!=allStatus2Taus.end(); ++iTau) {

    // look for all status 1 (stable) descendents 
    GenParticleRefVector descendents;
    findDescendents( *iTau, descendents, 1);
    
    // loop on descendents, and take all except neutrinos
  }
}


DEFINE_FWK_MODULE( TauGenJetProducer );
