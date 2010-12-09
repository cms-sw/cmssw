#include "PhysicsTools/RecoAlgos/interface/MassKinFitterCandProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Candidate/interface/Candidate.h"

MassKinFitterCandProducer::MassKinFitterCandProducer(const edm::ParameterSet & cfg, CandMassKinFitter * f) :
  src_(cfg.getParameter<edm::InputTag>("src")),
  fitter_(f) {
  if(f == 0) fitter_.reset(new CandMassKinFitter(cfg.getParameter<double>("mass")));
  produces<reco::CandidateCollection>();
}

void MassKinFitterCandProducer::produce( edm::Event & evt, const edm::EventSetup & es ) {
  using namespace edm; 
  using namespace reco;
  Handle<CandidateCollection> cands;
  evt.getByLabel(src_, cands);
  std::auto_ptr<CandidateCollection> refitted( new CandidateCollection );
  for( CandidateCollection::const_iterator c = cands->begin(); c != cands->end(); ++ c ) {
    Candidate * clone = c->clone();
    fitter_->set( * clone );
    refitted->push_back( clone );
  }
  evt.put( refitted );
}

