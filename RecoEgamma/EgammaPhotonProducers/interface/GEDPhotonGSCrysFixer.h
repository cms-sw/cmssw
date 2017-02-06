#ifndef RecoEgamma_EgammaPhotonProducers_GEDPhotonGSCrysFixer_h
#define RecoEgamma_EgammaPhotonProducers_GEDPhotonGSCrysFixer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCoreFwd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "RecoEgamma/EgammaPhotonAlgos/interface/PhotonEnergyCorrector.h"

#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

class GEDPhotonGSCrysFixer : public edm::stream::EDProducer<> {
 public:

  GEDPhotonGSCrysFixer (const edm::ParameterSet&);
  ~GEDPhotonGSCrysFixer();

  void beginLuminosityBlock(edm::LuminosityBlock const&,
                            edm::EventSetup const&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;

 private:
  template<typename T>
  void
  getToken(edm::EDGetTokenT<T>& token, edm::ParameterSet const& pset, std::string const& label, std::string const& instance = "")
  {
    auto tag(pset.getParameter<edm::InputTag>(label));
    if (!instance.empty())
      tag = edm::InputTag(tag.label(), instance, tag.process());

    token = consumes<T>(tag);
  }

  template<typename T>
  edm::Handle<T>
  getHandle(edm::Event const& iEvent, edm::EDGetTokenT<T> const& token, std::string const& name)
  {
    edm::Handle<T> handle;
    if (!iEvent.getByToken(token, handle))
      throw cms::Exception("ProductNotFound") << name;

    return handle;
  }

  typedef edm::View<reco::Photon> PhotonView;
  typedef edm::ValueMap<reco::SuperClusterRef> SCRefMap;

  edm::EDGetTokenT<PhotonView> inputPhotonsToken_;
  edm::EDGetTokenT<reco::PhotonCoreCollection> newCoresToken_;
  edm::EDGetTokenT<SCRefMap> newCoresToOldSCMapToken_;
  edm::EDGetTokenT<EcalRecHitCollection> ebHitsToken_;
  edm::EDGetTokenT<reco::VertexCollection> verticesToken_;

  PhotonEnergyCorrector energyCorrector_;

  CaloTopology const* topology_{0};
  CaloGeometry const* geometry_{0};
};

#endif
