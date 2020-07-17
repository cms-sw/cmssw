#include <memory>



#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
        #include "PhysicsTools/RecoAlgos/interface/MassKinFitterCandProducer.h"

MassKinFitterCandProducer::MassKinFitterCandProducer(const edm::ParameterSet& cfg, CandMassKinFitter* f)
    : srcToken_(consumes<reco::CandidateCollection>(cfg.getParameter<edm::InputTag>("src"))), fitter_(f) {
  if (f == nullptr)
    fitter_ = std::make_unique<CandMassKinFitter>(cfg.getParameter<double>("mass"));
  produces<reco::CandidateCollection>();
}

void MassKinFitterCandProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  using namespace edm;
  using namespace reco;
  Handle<CandidateCollection> cands;
  evt.getByToken(srcToken_, cands);
  auto refitted = std::make_unique<CandidateCollection>();
  for (CandidateCollection::const_iterator c = cands->begin(); c != cands->end(); ++c) {
    Candidate* clone = c->clone();
    fitter_->set(*clone);
    refitted->push_back(clone);
  }
  evt.put(std::move(refitted));
}
