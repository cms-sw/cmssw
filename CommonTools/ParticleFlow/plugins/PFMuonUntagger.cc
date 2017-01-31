/// Take a Muon collection and one or more lists of bad muons to un-PF-tag
//  Produce:
//      a Muon collection in which those muons are no longer PF
//      a Muon collection with the original copy of only the muons that were changed
//      a ValueMap<int> keyed to the new collection, with the old PF id
//      Association<MuonCollection> that maps old to new and vice-versa,
//      and from bad to new and vice-versa

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/MuonReco/interface/Muon.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Association.h"
#include <iostream>

class PFMuonUntagger : public edm::stream::EDProducer<> {
    public:
        PFMuonUntagger(const edm::ParameterSet&);
        ~PFMuonUntagger() {};

        void produce(edm::Event&, const edm::EventSetup&);

    private:
        edm::EDGetTokenT<std::vector<reco::Muon> > muons_;
        std::vector<edm::EDGetTokenT<reco::CandidateView>> badmuons_;

        template<typename H1, typename H2>
        void writeAssociation(edm::Event &out, const H1 &from, const H2 &to, const std::vector<int> indices, const std::string &name) {
            typedef edm::Association<std::vector<reco::Muon>> AssoMap;
            std::unique_ptr<AssoMap> assomap(new AssoMap(to));
            typename AssoMap::Filler filler(*assomap);
            filler.insert(from, indices.begin(), indices.end());
            filler.fill();
            out.put(std::move(assomap), name);
        }

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

PFMuonUntagger::PFMuonUntagger(const edm::ParameterSet &iConfig) :
    muons_(consumes<std::vector<reco::Muon>>(iConfig.getParameter<edm::InputTag>("muons")))
{
    for (const auto & src : iConfig.getParameter<std::vector<edm::InputTag>>("badmuons")) {
        badmuons_.push_back(consumes<reco::CandidateView>(src));
    }

    produces<std::vector<reco::Muon>>();
    produces<edm::ValueMap<int>>("oldPF");
    produces<edm::Association<std::vector<reco::Muon>>>("newToOld");
    produces<edm::Association<std::vector<reco::Muon>>>("oldToNew");
    produces<std::vector<reco::Muon>>("bad");
    produces<edm::Association<std::vector<reco::Muon>>>("badToNew");
    produces<edm::Association<std::vector<reco::Muon>>>("newToBad");
}

void PFMuonUntagger::produce(edm::Event &iEvent, const edm::EventSetup&)
{
    edm::Handle<std::vector<reco::Muon>> muons;
    iEvent.getByToken(muons_, muons);

    unsigned int n = muons->size();
    std::unique_ptr<std::vector<reco::Muon>> copy(new std::vector<reco::Muon>(*muons));
    std::vector<int> oldPF(n); 
    std::vector<int> dummyIndices(n); 
    for (unsigned int i = 0; i < n; ++i) {
        oldPF[i] = (*copy)[i].isPFMuon();
        dummyIndices[i] = i;
    }

    edm::Handle<reco::CandidateView> badmuons;
    for (const auto & tag : badmuons_) {
        iEvent.getByToken(tag, badmuons);
        for (unsigned int j = 0, nj = badmuons->size(); j < nj; ++j) {
            reco::CandidatePtr p = badmuons->ptrAt(j);
            if (p.isNonnull() && p.id() == muons.id()) {
                reco::Muon &mu = (*copy)[p.key()];
                mu.setType(mu.type() & ~reco::Muon::PFMuon);
            }
        }
    }

    std::unique_ptr<std::vector<reco::Muon>> bad(new std::vector<reco::Muon>());
    std::vector<int> good2bad(n,-1), bad2good;
    for (unsigned int i = 0; i < n; ++i) {
        const reco::Muon &mu = (*copy)[i];
        if (oldPF[i] != mu.isPFMuon()) {
            bad->push_back((*muons)[i]);
            bad2good.push_back(i);
            good2bad[i] = (bad->size()-1);
        }
    }
    
    // Now we put things in the event
    edm::OrphanHandle<std::vector<reco::Muon>> newmu = iEvent.put(std::move(copy));
    edm::OrphanHandle<std::vector<reco::Muon>> badmu = iEvent.put(std::move(bad), "bad");

    // Now we create the associations
    writeAssociation(iEvent, newmu, muons, dummyIndices, "newToOld");
    writeAssociation(iEvent, muons, newmu, dummyIndices, "oldToNew");
    writeAssociation(iEvent, newmu, badmu, good2bad, "newToBad");
    writeAssociation(iEvent, badmu, newmu, bad2good, "badToNew");

    // Now we create the valuemap
    writeValueMap(iEvent, newmu, oldPF, "oldPF");
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFMuonUntagger);
