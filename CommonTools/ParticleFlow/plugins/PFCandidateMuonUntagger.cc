/// Take:
//    - a PF candidate collection (which uses the old muons)
//    - a map from the old muons to the new muons (in which some muons have been un-tagged and so are no longer PF muons) 
//      format: edm::Association<std::vector<reco::Muon>>
//  Produce:
//    - a new PFCandidate collection using the new muons, and in which the muons that have been un-tagged are removed
//    - a second PFCandidate collection with just those discarded muons
//    - a ValueMap<reco::PFCandidateRef> that maps the old to the new, and vice-versa

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Association.h"
#include <iostream>

class PFCandidateMuonUntagger : public edm::stream::EDProducer<> {
    public:
        PFCandidateMuonUntagger(const edm::ParameterSet&);
        ~PFCandidateMuonUntagger() {};

        void produce(edm::Event&, const edm::EventSetup&);

    private:
        edm::EDGetTokenT<std::vector<reco::PFCandidate> > pfcandidates_;
        edm::EDGetTokenT<edm::Association<std::vector<reco::Muon>>> oldToNewMuons_;

        template<typename H1>
        void writeValueMap(edm::Event &out, const H1 &from, const std::vector<int> values, const std::string &name) {
            typedef edm::ValueMap<int> IntMap;
            std::unique_ptr<IntMap> intmap(new IntMap());
            typename IntMap::Filler filler(*intmap);
            filler.insert(from, values.begin(), values.end());
            filler.fill();
            out.put(std::move(intmap), name);
        }

};

PFCandidateMuonUntagger::PFCandidateMuonUntagger(const edm::ParameterSet &iConfig) :
    pfcandidates_(consumes<std::vector<reco::PFCandidate>>(iConfig.getParameter<edm::InputTag>("pfcandidates"))),
    oldToNewMuons_(consumes<edm::Association<std::vector<reco::Muon>>>(iConfig.getParameter<edm::InputTag>("oldToNewMuons")))
{
    produces<std::vector<reco::PFCandidate>>();
    produces<std::vector<reco::PFCandidate>>("discarded");
    produces<edm::ValueMap<reco::PFCandidateRef>>();
}

void PFCandidateMuonUntagger::produce(edm::Event &iEvent, const edm::EventSetup&)
{
    edm::Handle<edm::Association<std::vector<reco::Muon>>> oldToNewMuons;
    iEvent.getByToken(oldToNewMuons_, oldToNewMuons);

    edm::Handle<std::vector<reco::PFCandidate>> pfcandidates;
    iEvent.getByToken(pfcandidates_, pfcandidates);

    int n = pfcandidates->size();
    std::unique_ptr<std::vector<reco::PFCandidate>> copy(new std::vector<reco::PFCandidate>());
    std::unique_ptr<std::vector<reco::PFCandidate>> discarded(new std::vector<reco::PFCandidate>());
    copy->reserve(n); 
    std::vector<int> oldToNew(n), newToOld, badToOld; 
    newToOld.reserve(n);

    int i = -1;
    for (const reco::PFCandidate &pf : *pfcandidates) {
        ++i;
        if (pf.muonRef().isNonnull()) {
            reco::MuonRef newRef = (*oldToNewMuons)[pf.muonRef()];
            if (abs(pf.pdgId()) == 13 && !newRef->isPFMuon()) { // was untagging
                discarded->push_back(pf);
                oldToNew[i] = (-discarded->size());
                badToOld.push_back(i);
                discarded->back().setMuonRef(newRef);
            } else {
                copy->push_back(pf);
                oldToNew[i] = (copy->size());
                newToOld.push_back(i);
                copy->back().setMuonRef(newRef);
            }
        } else {
            copy->push_back(pf);
            oldToNew[i] = (copy->size());
            newToOld.push_back(i);
        } 
    }

    // Now we put things in the event
    edm::OrphanHandle<std::vector<reco::PFCandidate>> newpf = iEvent.put(std::move(copy));
    edm::OrphanHandle<std::vector<reco::PFCandidate>> badpf = iEvent.put(std::move(discarded), "discarded");

    std::unique_ptr<edm::ValueMap<reco::PFCandidateRef>> pf2pf(new edm::ValueMap<reco::PFCandidateRef>());
    edm::ValueMap<reco::PFCandidateRef>::Filler filler(*pf2pf);
    std::vector<reco::PFCandidateRef> refs; refs.reserve(n);
    // old to new
    for (i = 0; i < n; ++i) {
        if (oldToNew[i] > 0) {
            refs.push_back(reco::PFCandidateRef(newpf, oldToNew[i]-1));
        } else {
            refs.push_back(reco::PFCandidateRef(badpf,-oldToNew[i]-1));
        }
    }
    filler.insert(pfcandidates, refs.begin(), refs.end());
    // new good to old
    refs.clear();
    for (int i : newToOld) {
        refs.push_back(reco::PFCandidateRef(pfcandidates,i));
    }
    filler.insert(newpf, refs.begin(), refs.end());
    // new bad to old
    refs.clear();
    for (int i : badToOld) {
        refs.push_back(reco::PFCandidateRef(pfcandidates,i));
    }
    filler.insert(badpf, refs.begin(), refs.end());
    // done
    filler.fill();
    iEvent.put(std::move(pf2pf));
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFCandidateMuonUntagger);
