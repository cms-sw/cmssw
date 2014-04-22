#include <string>
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

namespace pat {
    class PATVertexSlimmer : public edm::EDProducer {
        public:
            explicit PATVertexSlimmer(const edm::ParameterSet&);
            ~PATVertexSlimmer();

            virtual void produce(edm::Event&, const edm::EventSetup&);
        private:
            edm::EDGetTokenT<std::vector<reco::Vertex> > src_;
    };
}

pat::PATVertexSlimmer::PATVertexSlimmer(const edm::ParameterSet& iConfig) :
    src_(consumes<std::vector<reco::Vertex> >(iConfig.getParameter<edm::InputTag>("src")))
{
  produces<std::vector<reco::Vertex> >();
}

pat::PATVertexSlimmer::~PATVertexSlimmer() {}

void pat::PATVertexSlimmer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    edm::Handle<std::vector<reco::Vertex> > vertices;
    iEvent.getByToken(src_, vertices);
    std::auto_ptr<std::vector<reco::Vertex> > outPtr(new std::vector<reco::Vertex>());

    outPtr->reserve(vertices->size());
    for (unsigned int i = 0, n = vertices->size(); i < n; ++i) { 
        const reco::Vertex &v = (*vertices)[i];
        outPtr->push_back(reco::Vertex(v.position(), v.error(), v.chi2(), v.ndof(), 0));
    }

    iEvent.put(outPtr);
}

using pat::PATVertexSlimmer;
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATVertexSlimmer);
