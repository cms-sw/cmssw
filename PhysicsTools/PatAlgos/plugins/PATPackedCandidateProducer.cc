#include <string>

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/Common/interface/Association.h"
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
            edm::EDGetTokenT<reco::PFCandidateFwdPtrVector>  CandsFromPVLoose_;
            edm::EDGetTokenT<reco::PFCandidateFwdPtrVector>  CandsFromPVTight_;
    };
}

pat::PATPackedCandidateProducer::PATPackedCandidateProducer(const edm::ParameterSet& iConfig) :
  Cands_(consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("inputCollection"))),
  CandsFromPVLoose_(consumes<reco::PFCandidateFwdPtrVector>(iConfig.getParameter<edm::InputTag>("inputCollectionFromPVLoose"))),
  CandsFromPVTight_(consumes<reco::PFCandidateFwdPtrVector>(iConfig.getParameter<edm::InputTag>("inputCollectionFromPVTight")))
{
  produces< std::vector<pat::PackedCandidate> > ();
  produces< edm::Association<pat::PackedCandidateCollection> > ();
  produces< edm::Association<reco::PFCandidateCollection> > ();
}

pat::PATPackedCandidateProducer::~PATPackedCandidateProducer() {}

void pat::PATPackedCandidateProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

    edm::Handle<reco::PFCandidateCollection> cands;
    iEvent.getByToken( Cands_, cands );
    std::vector<reco::Candidate>::const_iterator cand;

    edm::Handle<reco::PFCandidateFwdPtrVector> candsFromPVLoose;
    iEvent.getByToken( CandsFromPVLoose_, candsFromPVLoose );
    edm::Handle<reco::PFCandidateFwdPtrVector> candsFromPVTight;
    iEvent.getByToken( CandsFromPVTight_, candsFromPVTight );

    std::vector<pat::PackedCandidate::PVAssoc> fromPV(cands->size(), pat::PackedCandidate::NoPV);
    for (const reco::PFCandidateFwdPtr &ptr : *candsFromPVLoose) {
        if (ptr.ptr().id() == cands.id()) {
            fromPV[ptr.ptr().key()]   = pat::PackedCandidate::PVLoose;
        } else if (ptr.backPtr().id() == cands.id()) {
            fromPV[ptr.backPtr().key()] = pat::PackedCandidate::PVLoose;
        } else {
            throw cms::Exception("Configuration", "The elements from 'inputCollectionFromPVLoose' don't point to 'inputCollection'\n");
        }
    }
    for (const reco::PFCandidateFwdPtr &ptr : *candsFromPVTight) {
        if (ptr.ptr().id() == cands.id()) {
            fromPV[ptr.ptr().key()]   = pat::PackedCandidate::PVTight;
        } else if (ptr.backPtr().id() == cands.id()) {
            fromPV[ptr.backPtr().key()] = pat::PackedCandidate::PVTight;
        } else {
            throw cms::Exception("Configuration", "The elements from 'inputCollectionFromPVTight' don't point to 'inputCollection'\n");
        }
    }


    std::auto_ptr< std::vector<pat::PackedCandidate> > outPtrP( new std::vector<pat::PackedCandidate> );
    std::vector<int> mapping(cands->size());

    for(unsigned int ic=0, nc = cands->size(); ic < nc; ++ic) {
        const reco::Candidate &cand=(*cands)[ic];

        math::XYZPoint vtx = cand.vertex();
        outPtrP->push_back( pat::PackedCandidate(cand.polarP4(), (cand.charge() ? vtx : math::XYZPoint()), cand.pdgId(), fromPV[ic]));

        mapping[ic] = ic; // trivial at the moment!
    }

    edm::OrphanHandle<pat::PackedCandidateCollection> oh = iEvent.put( outPtrP );

    // now build the two maps
    std::auto_ptr<edm::Association<pat::PackedCandidateCollection> > pf2pc(new edm::Association<pat::PackedCandidateCollection>(oh   ));
    std::auto_ptr<edm::Association<reco::PFCandidateCollection   > > pc2pf(new edm::Association<reco::PFCandidateCollection   >(cands));
    edm::Association<pat::PackedCandidateCollection>::Filler pf2pcFiller(*pf2pc);
    edm::Association<reco::PFCandidateCollection   >::Filler pc2pfFiller(*pc2pf);
    pf2pcFiller.insert(cands, mapping.begin(), mapping.end());
    pc2pfFiller.insert(oh   , mapping.begin(), mapping.end());
    pf2pcFiller.fill();
    pc2pfFiller.fill();
    iEvent.put(pf2pc);
    iEvent.put(pc2pf);

}


using pat::PATPackedCandidateProducer;
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATPackedCandidateProducer);
