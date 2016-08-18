/**
  \class    pat::PATVertexAssociationProducer PATVertexAssociationProducer.h "PhysicsTools/PatAlgos/interface/PATVertexAssociationProducer.h"
  \brief    Produces VertexAssociation and a ValueMap to the originating
            reco jets

   The PATVertexAssociationProducer produces a set of vertex associations for one or more
   collection of Candidates, and saves them in a ValueMap in the event.

   These can be retrieved in PAT Layer 1 to be embedded in PAT Objects

  \author   Giovanni Petrucciani
  \version  $Id: VertexAssociationProducer.cc,v 1.2 2010/02/20 21:00:29 wmtan Exp $
*/


#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/PatCandidates/interface/Vertexing.h"
#include "PhysicsTools/PatAlgos/interface/VertexingHelper.h"


namespace pat {

  class PATVertexAssociationProducer : public edm::stream::EDProducer<> {

    typedef edm::ValueMap<pat::VertexAssociation> VertexAssociationMap;

    public:

      explicit PATVertexAssociationProducer(const edm::ParameterSet & iConfig);
      ~PATVertexAssociationProducer();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup& iSetup) override;

    private:
      typedef std::vector<edm::InputTag> VInputTag;
      // configurables
      std::vector<edm::InputTag>   particles_;
      std::vector<edm::EDGetTokenT<edm::View<reco::Candidate> > > particlesTokens_;
      pat::helper::VertexingHelper vertexing_;

  };

}

using pat::PATVertexAssociationProducer;

PATVertexAssociationProducer::PATVertexAssociationProducer(const edm::ParameterSet& iConfig) :
  particles_( iConfig.existsAs<VInputTag>("candidates") ?       // if it's a VInputTag
                iConfig.getParameter<VInputTag>("candidates") :
                VInputTag(1, iConfig.getParameter<edm::InputTag>("candidates")) ),
  vertexing_(iConfig, consumesCollector())
{
  for (VInputTag::const_iterator it = particles_.begin(), end = particles_.end(); it != end; ++it) {
    particlesTokens_.push_back( consumes<edm::View<reco::Candidate> >( *it ) );
  }
    produces<VertexAssociationMap>();
}


PATVertexAssociationProducer::~PATVertexAssociationProducer() {
}


void PATVertexAssociationProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
  using namespace edm; using namespace std;
  // read in vertices and EventSetup
  vertexing_.newEvent(iEvent, iSetup);

  // prepare room and tools for output
  auto result = std::make_unique<VertexAssociationMap>();
  VertexAssociationMap::Filler filler(*result);
  vector<pat::VertexAssociation> assos;

  // loop on input tags
  for (std::vector<edm::EDGetTokenT<edm::View<reco::Candidate> > >::const_iterator it = particlesTokens_.begin(), end = particlesTokens_.end(); it != end; ++it) {
      // read candidates
      Handle<View<reco::Candidate> > cands;
      iEvent.getByToken(*it, cands);
      assos.clear(); assos.reserve(cands->size());
      // loop on candidates
      for (size_t i = 0, n = cands->size(); i < n; ++i) {
        assos.push_back( vertexing_(cands->refAt(i)) );
      }
      // insert into ValueMap
      filler.insert(cands, assos.begin(), assos.end());
  }

  // do the real filling
  filler.fill();

  // put our produced stuff in the event
  iEvent.put(std::move(result));
}


#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATVertexAssociationProducer);
