#include <string>

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/MuonReco/interface/Muon.h"

namespace pat {
    class PATPackedCandidateProducer : public edm::EDProducer {
        public:
            explicit PATPackedCandidateProducer(const edm::ParameterSet&);
            ~PATPackedCandidateProducer();

            virtual void produce(edm::Event&, const edm::EventSetup&);

        private:
            edm::EDGetTokenT<reco::PFCandidateCollection>    Cands_;
            edm::EDGetTokenT<reco::PFCandidateFwdPtrVector>  CandsFromPV_;
    };
}

pat::PATPackedCandidateProducer::PATPackedCandidateProducer(const edm::ParameterSet& iConfig) :
  Cands_(consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("inputCollection"))),
  CandsFromPV_(consumes<reco::PFCandidateFwdPtrVector>(iConfig.getParameter<edm::InputTag>("inputCollectionFromPV")))
{
  produces< std::vector<pat::PackedCandidate> > ();
}

pat::PATPackedCandidateProducer::~PATPackedCandidateProducer() {}

void pat::PATPackedCandidateProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

    edm::Handle<reco::PFCandidateCollection> cands;
    iEvent.getByToken( Cands_, cands );
    std::vector<reco::Candidate>::const_iterator cand;

    edm::Handle<reco::PFCandidateFwdPtrVector> candsFromPV;
    iEvent.getByToken( CandsFromPV_, candsFromPV );

    std::vector<bool> fromPV(cands->size(), false);
    for (const reco::PFCandidateFwdPtr &ptr : *candsFromPV) {
        if (ptr.ptr().id() == cands.id()) {
            fromPV[ptr.ptr().key()] = true;
        } else if (ptr.backPtr().id() == cands.id()) {
            fromPV[ptr.backPtr().key()] = true;
        } else {
            throw cms::Exception("Configuration", "The elements from 'inputCollectionFromPV' don't point to 'inputCollection'\n");
        }
    }

    std::auto_ptr< std::vector<pat::PackedCandidate> > outPtrP( new std::vector<pat::PackedCandidate> );

    for(unsigned int ic=0, nc = cands->size(); ic < nc; ++ic) {
        const reco::Candidate &cand=(*cands)[ic];

        math::XYZPoint vtx = cand.vertex();
        outPtrP->push_back( pat::PackedCandidate(cand.polarP4(), (cand.charge() ? vtx : math::XYZPoint()), cand.pdgId(), fromPV[ic]));

    }

    iEvent.put( outPtrP );
}


using pat::PATPackedCandidateProducer;
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATPackedCandidateProducer);
