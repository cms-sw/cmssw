#ifndef RecoEgamma_EgammaPhotonProducers_GEDPhotonCoreGSCrysFixer_h
#define RecoEgamma_EgammaPhotonProducers_GEDPhotonCoreGSCrysFixer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCoreFwd.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

class GEDPhotonCoreGSCrysFixer : public edm::stream::EDProducer<> {
 public:

  GEDPhotonCoreGSCrysFixer (const edm::ParameterSet&);
  ~GEDPhotonCoreGSCrysFixer();

  void produce(edm::Event&, const edm::EventSetup&) override;

 private:
  typedef edm::ValueMap<reco::SuperClusterRef> SCRefMap;
  typedef edm::ValueMap<reco::ConversionRef> ConvRefMap;

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
  getHandle(edm::Event const& _event, edm::EDGetTokenT<T> const& token, std::string const& name)
  {
    edm::Handle<T> handle;
    if (!_event.getByToken(token, handle))
      throw cms::Exception("ProductNotFound") << name;

    return handle;
  }

  edm::EDGetTokenT<reco::PhotonCoreCollection> inputCoresToken_;
  edm::EDGetTokenT<reco::SuperClusterCollection> refinedSCsToken_; // new
  edm::EDGetTokenT<SCRefMap> refinedSCMapToken_; // new->old
  edm::EDGetTokenT<reco::SuperClusterCollection> ebSCsToken_; // new
  edm::EDGetTokenT<SCRefMap> ebSCMapToken_; // new->old
  edm::EDGetTokenT<reco::SuperClusterCollection> eeSCsToken_; // new
  edm::EDGetTokenT<SCRefMap> eeSCMapToken_; // new->old
  edm::EDGetTokenT<reco::ConversionCollection> convsToken_; // new
  edm::EDGetTokenT<ConvRefMap> convMapToken_; // new->old
  edm::EDGetTokenT<reco::ConversionCollection> singleLegConvsToken_; // new
  edm::EDGetTokenT<ConvRefMap> singleLegConvMapToken_; // new->old
};

#endif
