/**
  \class    pat::PATTauSlimmer PATTauSlimmer.h "PhysicsTools/PatAlgos/interface/PATTauSlimmer.h"
  \brief    Slimmer of PAT Taus 
*/


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/PatCandidates/interface/Tau.h"

namespace pat {

  class PATTauSlimmer : public edm::EDProducer {
    public:
      explicit PATTauSlimmer(const edm::ParameterSet & iConfig);
      virtual ~PATTauSlimmer() { }

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

    private:
      edm::EDGetTokenT<edm::View<pat::Tau> > src_;

  };

} // namespace

pat::PATTauSlimmer::PATTauSlimmer(const edm::ParameterSet & iConfig) :
    src_(consumes<edm::View<pat::Tau> >(iConfig.getParameter<edm::InputTag>("src")))
{
    produces<std::vector<pat::Tau> >();
}

void 
pat::PATTauSlimmer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
    using namespace edm;
    using namespace std;

    Handle<View<pat::Tau> >      src;
    iEvent.getByToken(src_, src);

    auto_ptr<vector<pat::Tau> >  out(new vector<pat::Tau>());
    out->reserve(src->size());

    for (View<pat::Tau>::const_iterator it = src->begin(), ed = src->end(); it != ed; ++it) {
        out->push_back(*it);
        //pat::Tau & tau = out->back();
    }

    iEvent.put(out);
}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(PATTauSlimmer);
