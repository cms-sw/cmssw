#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"

namespace pat {
  

  template<typename T>
  class LeptonUpdater : public edm::global::EDProducer<> {

    public:

      explicit LeptonUpdater(const edm::ParameterSet & iConfig) :
            src_(consumes<std::vector<T>>(iConfig.getParameter<edm::InputTag>("src"))),
            vertices_(consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("vertices")))
        {
            produces<std::vector<T>>();
        }

      ~LeptonUpdater() override {}

      void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override ;

      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
          edm::ParameterSetDescription desc;
          desc.add<edm::InputTag>("src")->setComment("Lepton collection");
          desc.add<edm::InputTag>("vertices")->setComment("Vertex collection");
          if (typeid(T) == typeid(pat::Muon)) descriptions.add("muonsUpdated", desc);
          else if (typeid(T) == typeid(pat::Electron)) descriptions.add("electronsUpdated", desc);
      }

      void setDZ(T & lep, const reco::Vertex & pv) const {}

    private:
      // configurables
      edm::EDGetTokenT<std::vector<T>> src_;
      edm::EDGetTokenT<std::vector<reco::Vertex>> vertices_;
  };

  // must do the specialization within the namespace otherwise gcc complains
  //
  template<>
  void LeptonUpdater<pat::Electron>::setDZ(pat::Electron & anElectron, const reco::Vertex & pv) const {
      auto track = anElectron.gsfTrack();
      anElectron.setDB( track->dz(pv.position()), std::hypot(track->dzError(), pv.zError()), pat::Electron::PVDZ );
  }
  
  template<>
  void LeptonUpdater<pat::Muon>::setDZ(pat::Muon & aMuon, const reco::Vertex & pv) const {
      auto track = aMuon.muonBestTrack();
      aMuon.setDB( track->dz(pv.position()), std::hypot(track->dzError(), pv.zError()), pat::Muon::PVDZ );
  }

} // namespace

template<typename T>
void pat::LeptonUpdater<T>::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
    edm::Handle<std::vector<T>> src;
    iEvent.getByToken(src_, src);

    edm::Handle<std::vector<reco::Vertex>> vertices;
    iEvent.getByToken(vertices_, vertices);
    const reco::Vertex & pv = vertices->front();

    std::unique_ptr<std::vector<T>> out(new std::vector<T>(*src));

    for (unsigned int i = 0, n = src->size(); i < n; ++i) {
        T & lep = (*out)[i];
        setDZ(lep, pv);
    }

    iEvent.put(std::move(out));
}



typedef pat::LeptonUpdater<pat::Electron> PATElectronUpdater;
typedef pat::LeptonUpdater<pat::Muon> PATMuonUpdater;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATElectronUpdater);
DEFINE_FWK_MODULE(PATMuonUpdater);
