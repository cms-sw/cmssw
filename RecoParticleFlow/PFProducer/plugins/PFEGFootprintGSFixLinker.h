#ifndef RecoParticleFlow_PFProducer_PFEGFootprintGSFixLinker_h
#define RecoParticleFlow_PFProducer_PFEGFootprintGSFixLinker_h

/** \class PFEGFootprintGSFixLinker
 *  Relink GS-fixed e/gamma objects
 *  \author Y. Iiyama (MIT)
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include <string>

class PFEGFootprintGSFixLinker : public edm::stream::EDProducer<> {
 public:
  explicit PFEGFootprintGSFixLinker(const edm::ParameterSet&);
  ~PFEGFootprintGSFixLinker();
  
  void produce(edm::Event&, const edm::EventSetup&) override;

  template<typename T>
  void
  getToken(edm::EDGetTokenT<T>& token, edm::ParameterSet const& pset, std::string const& label)
  {
    token = consumes<T>(pset.getParameter<edm::InputTag>(label));
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

 private:
  typedef edm::ValueMap<reco::GsfElectronRef> ElectronRefMap;
  typedef edm::ValueMap<reco::PhotonRef> PhotonRefMap;
  typedef std::vector<reco::PFCandidateRef> Footprint;
  typedef edm::ValueMap<Footprint> FootprintMap;

  edm::EDGetTokenT<reco::PFCandidateCollection> newCandidatesToken_;
  edm::EDGetTokenT<reco::GsfElectronCollection> newElectronsToken_;
  edm::EDGetTokenT<reco::PhotonCollection> newPhotonsToken_;
  // new -> old map
  edm::EDGetTokenT<ElectronRefMap> electronMapToken_;
  edm::EDGetTokenT<PhotonRefMap> photonMapToken_;
  // e/g -> footprint map
  edm::EDGetTokenT<FootprintMap> electronFootprintMapToken_;
  edm::EDGetTokenT<FootprintMap> photonFootprintMapToken_;

  std::string electronsMapName_;
  std::string photonsMapName_;
};

#endif
