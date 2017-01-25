#ifndef RecoEgamma_EgammaPhotonProducers_ConversionGSCrysFixer_h
#define RecoEgamma_EgammaPhotonProducers_ConversionGSCrysFixer_h
/** \class ConversionGSCrysFixer
 **  
 **
 **  \author Yutaro Iiyama, MIT
 **
 ***/

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

class ConversionGSCrysFixer : public edm::stream::EDProducer<> {
 public:

  ConversionGSCrysFixer (const edm::ParameterSet&);
  ~ConversionGSCrysFixer();

  void produce(edm::Event&, const edm::EventSetup&) override;

 private:
  typedef edm::ValueMap<reco::SuperClusterRef> SCRefMap;

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

  edm::EDGetTokenT<reco::ConversionCollection> inputConvsToken_;
  //  edm::EDGetTokenT<reco::ConversionCollection> inputSingleLegConvsToken_;
  //  edm::EDGetTokenT<reco::SuperClusterCollection> refinedSCsToken_; // new
  //  edm::EDGetTokenT<SCRefMap> refinedSCMapToken_; // new->old
  edm::EDGetTokenT<reco::SuperClusterCollection> ebSCsToken_; // new
  edm::EDGetTokenT<SCRefMap> ebSCMapToken_; // new->old
  edm::EDGetTokenT<reco::SuperClusterCollection> eeSCsToken_; // new
  edm::EDGetTokenT<SCRefMap> eeSCMapToken_; // new->old
};

#endif
