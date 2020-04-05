/**
  Take as input:
     - the electron and photon collections
     - the electron and photon pf maps (edm::ValueMap<std::vector<reco::PFCandidateRef>>) pointing to old PF candidates
  Produce as output:
     - the electron and photon collections as VertexCompositePtrCandidates
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"

class PFEGammaToCandidate : public edm::global::EDProducer<> {
public:
  explicit PFEGammaToCandidate(const edm::ParameterSet &iConfig);
  ~PFEGammaToCandidate() override {}

  void produce(edm::StreamID iID, edm::Event &iEvent, const edm::EventSetup &iSetup) const override;

private:
  edm::EDGetTokenT<edm::View<pat::Photon>> photons_;
  edm::EDGetTokenT<edm::View<pat::Electron>> electrons_;
  edm::EDGetTokenT<edm::ValueMap<std::vector<reco::PFCandidateRef>>> photon2pf_, electron2pf_;

  template <typename T>
  void run(edm::Event &iEvent,
           const edm::EDGetTokenT<edm::View<T>> &colltoken,
           const edm::EDGetTokenT<edm::ValueMap<std::vector<reco::PFCandidateRef>>> &oldmaptoken,
           const std::string &name) const {
    edm::Handle<edm::View<T>> handle;
    iEvent.getByToken(colltoken, handle);

    edm::Handle<edm::ValueMap<std::vector<reco::PFCandidateRef>>> oldmap;
    iEvent.getByToken(oldmaptoken, oldmap);

    auto result = std::make_unique<std::vector<reco::VertexCompositePtrCandidate>>();
    for (unsigned int i = 0, n = handle->size(); i < n; ++i) {
      auto &obj = (*handle)[i];
      result->push_back(reco::VertexCompositePtrCandidate(obj));
      for (reco::PFCandidateRef pfRef : (*oldmap)[obj.originalObjectRef()])
        result->back().addDaughter(refToPtr(pfRef));
    }
    iEvent.put(std::move(result), name);
  }
};

PFEGammaToCandidate::PFEGammaToCandidate(const edm::ParameterSet &iConfig)
    : photons_(consumes<edm::View<pat::Photon>>(iConfig.getParameter<edm::InputTag>("photons"))),
      electrons_(consumes<edm::View<pat::Electron>>(iConfig.getParameter<edm::InputTag>("electrons"))),
      photon2pf_(
          consumes<edm::ValueMap<std::vector<reco::PFCandidateRef>>>(iConfig.getParameter<edm::InputTag>("photon2pf"))),
      electron2pf_(consumes<edm::ValueMap<std::vector<reco::PFCandidateRef>>>(
          iConfig.getParameter<edm::InputTag>("electron2pf"))) {
  produces<std::vector<reco::VertexCompositePtrCandidate>>("photons");
  produces<std::vector<reco::VertexCompositePtrCandidate>>("electrons");
}

void PFEGammaToCandidate::produce(edm::StreamID iID, edm::Event &iEvent, const edm::EventSetup &iSetup) const {
  run<pat::Photon>(iEvent, photons_, photon2pf_, "photons");
  run<pat::Electron>(iEvent, electrons_, electron2pf_, "electrons");
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFEGammaToCandidate);
