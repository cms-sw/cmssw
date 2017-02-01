/**
  Take as input:
     - the electron and photon collections
     - the electron and photon pf maps (edm::ValueMap<std::vector<reco::PFCandidateRef>>) pointing to old PF candidates
     - one ValueMap<reco::PFCandidateRef> that maps old PF candidates into new PF candidates
  Produce as output:
     - the new electron and photon pf maps
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

class PFEGammaToCandidateRemapper : public edm::stream::EDProducer<> {
    public:
        explicit PFEGammaToCandidateRemapper(const edm::ParameterSet & iConfig);
        virtual ~PFEGammaToCandidateRemapper() { }

        virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

    private:
        edm::EDGetTokenT<std::vector<reco::Photon>> photons_;
        edm::EDGetTokenT<std::vector<reco::GsfElectron>> electrons_;
        edm::EDGetTokenT<edm::ValueMap<std::vector<reco::PFCandidateRef>>> photon2pf_, electron2pf_;
        edm::EDGetTokenT<edm::ValueMap<reco::PFCandidateRef>> pf2pf_;

        template<typename T>
        void run(edm::Event & iEvent, 
                const edm::EDGetTokenT<std::vector<T>> &colltoken, 
                const edm::EDGetTokenT<edm::ValueMap<std::vector<reco::PFCandidateRef>>> &oldmaptoken, 
                const edm::ValueMap<reco::PFCandidateRef> & pf2pf, 
                const std::string &name) {
            edm::Handle<std::vector<T>> handle;
            iEvent.getByToken(colltoken, handle);

            edm::Handle<edm::ValueMap<std::vector<reco::PFCandidateRef>>> oldmap;
            iEvent.getByToken(oldmaptoken, oldmap);

            std::vector<std::vector<reco::PFCandidateRef>> refs(handle->size());
            for (unsigned int i = 0, n = handle->size(); i < n; ++i) {
                edm::Ref<std::vector<T>> egRef(handle,i);
                for (reco::PFCandidateRef pfRef : (*oldmap)[egRef]) {
                    refs[i].push_back(pf2pf[pfRef]);
                }
            }

            std::unique_ptr<edm::ValueMap<std::vector<reco::PFCandidateRef>>> out(new edm::ValueMap<std::vector<reco::PFCandidateRef>>());
            edm::ValueMap<std::vector<reco::PFCandidateRef>>::Filler filler(*out);
            filler.insert(handle, refs.begin(), refs.end());
            filler.fill();
            iEvent.put(std::move(out), name);
        }
};


PFEGammaToCandidateRemapper::PFEGammaToCandidateRemapper(const edm::ParameterSet & iConfig) :
    photons_(consumes<std::vector<reco::Photon>>(iConfig.getParameter<edm::InputTag>("photons"))),
    electrons_(consumes<std::vector<reco::GsfElectron>>(iConfig.getParameter<edm::InputTag>("electrons"))),
    photon2pf_(consumes<edm::ValueMap<std::vector<reco::PFCandidateRef>>>(iConfig.getParameter<edm::InputTag>("photon2pf"))),
    electron2pf_(consumes<edm::ValueMap<std::vector<reco::PFCandidateRef>>>(iConfig.getParameter<edm::InputTag>("electron2pf"))),
    pf2pf_(consumes<edm::ValueMap<reco::PFCandidateRef>>(iConfig.getParameter<edm::InputTag>("pf2pf")))
{
    produces<edm::ValueMap<std::vector<reco::PFCandidateRef>>>("photons");
    produces<edm::ValueMap<std::vector<reco::PFCandidateRef>>>("electrons");
}


void 
PFEGammaToCandidateRemapper::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
    edm::Handle<edm::ValueMap<reco::PFCandidateRef>> pf2pf;
    iEvent.getByToken(pf2pf_, pf2pf);
    run<reco::Photon>(iEvent, photons_, photon2pf_, *pf2pf, "photons");
    run<reco::GsfElectron>(iEvent, electrons_, electron2pf_, *pf2pf, "electrons");
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFEGammaToCandidateRemapper);
