#include <string>
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/Common/interface/RefToPtr.h"

namespace pat {
    class PATSecondaryVertexSlimmer : public edm::EDProducer {
        public:
            explicit PATSecondaryVertexSlimmer(const edm::ParameterSet&);
            ~PATSecondaryVertexSlimmer();

            virtual void produce(edm::Event&, const edm::EventSetup&);
        private:
            edm::EDGetTokenT<reco::VertexCompositePtrCandidateCollection> src_;
            edm::EDGetTokenT<std::vector<reco::Vertex> > srcLegacy_;
            edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection> > map_;
            edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection> > map2_;
    };
}

pat::PATSecondaryVertexSlimmer::PATSecondaryVertexSlimmer(const edm::ParameterSet& iConfig) :
    src_(mayConsume<reco::VertexCompositePtrCandidateCollection>(iConfig.getParameter<edm::InputTag>("src"))),
    srcLegacy_(mayConsume<std::vector<reco::Vertex> >(iConfig.getParameter<edm::InputTag>("src"))),
    map_(consumes<edm::Association<pat::PackedCandidateCollection> >(iConfig.getParameter<edm::InputTag>("packedPFCandidates"))),
    map2_(mayConsume<edm::Association<pat::PackedCandidateCollection> >(iConfig.existsAs<edm::InputTag>("lostTracksCandidates") ? iConfig.getParameter<edm::InputTag>("lostTracksCandidates") : edm::InputTag("lostTracks") ))
{
  produces< reco::VertexCompositePtrCandidateCollection >();
}

pat::PATSecondaryVertexSlimmer::~PATSecondaryVertexSlimmer() {}

void pat::PATSecondaryVertexSlimmer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

    std::auto_ptr<reco::VertexCompositePtrCandidateCollection> outPtr(new reco::VertexCompositePtrCandidateCollection);
 
    edm::Handle<reco::VertexCompositePtrCandidateCollection> candVertices;
    iEvent.getByToken(src_, candVertices);
        
    edm::Handle<edm::Association<pat::PackedCandidateCollection> > pf2pc;
    iEvent.getByToken(map_,pf2pc);

    // if reco::VertexCompositePtrCandidate secondary vertices are present
    if( candVertices.isValid() )
    {
        outPtr->reserve(candVertices->size());
        for (unsigned int i = 0, n = candVertices->size(); i < n; ++i) {

                reco::VertexCompositePtrCandidate v = (*candVertices)[i];

                std::vector<reco::CandidatePtr> daughters = v.daughterPtrVector();
                v.clearDaughters();

                for(std::vector<reco::CandidatePtr>::const_iterator it = daughters.begin(); it != daughters.end(); ++it) {

                    if((*pf2pc)[*it].isNonnull() && (*pf2pc)[*it]->numberOfHits() > 0)
                        v.addDaughter(reco::CandidatePtr(edm::refToPtr((*pf2pc)[*it]) ));
                }

                outPtr->push_back( v );
        }
    }
    // otherwise fallback to reco::Vertex secondary vertices
    else
    {
        edm::Handle<std::vector<reco::Vertex> > vertices;
        iEvent.getByToken(srcLegacy_, vertices);

        edm::Handle<edm::Association<pat::PackedCandidateCollection> > pf2pc2;
        iEvent.getByToken(map2_,pf2pc2);

        outPtr->reserve(vertices->size());
        for (unsigned int i = 0, n = vertices->size(); i < n; ++i) {
                const reco::Vertex &v = (*vertices)[i];
                outPtr->push_back(reco::VertexCompositePtrCandidate(0,v.p4(),v.position(), v.error(), v.chi2(), v.ndof()));

                for(reco::Vertex::trackRef_iterator  it=v.tracks_begin(); it != v.tracks_end(); it++) {
                        if(v.trackWeight(*it)>0.5) {
                                if((*pf2pc)[*it].isNonnull() && (*pf2pc)[*it]->numberOfHits() > 0) {
                                        outPtr->back().addDaughter(reco::CandidatePtr(edm::refToPtr((*pf2pc)[*it]) ));
                                }
                                else {
                                        if((*pf2pc2)[*it].isNonnull()) {
                                                outPtr->back().addDaughter(reco::CandidatePtr(edm::refToPtr((*pf2pc2)[*it]) ));
                                        }	
                                        else { std::cout << "HELPME" << std::endl;}	
                                }
                        }
                }
        }
    }

    iEvent.put(outPtr);
}

using pat::PATSecondaryVertexSlimmer;
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATSecondaryVertexSlimmer);
