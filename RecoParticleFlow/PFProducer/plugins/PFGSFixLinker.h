#ifndef RecoParticleFlow_PFProducer_PFGSFixLinker_h
#define RecoParticleFlow_PFProducer_PFGSFixLinker_h

/** \class PFGSFixLinker
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

class PFGSFixLinker : public edm::stream::EDProducer<> {
 public:
  explicit PFGSFixLinker(const edm::ParameterSet&);
  ~PFGSFixLinker();
  
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
  typedef edm::View<reco::PFCandidate> PFCandidateView;
  typedef edm::ValueMap<reco::GsfElectronRef> ElectronRefMap;
  typedef edm::ValueMap<reco::PhotonRef> PhotonRefMap;

  edm::EDGetTokenT<PFCandidateView> inputCandidatesToken_;
  edm::EDGetTokenT<reco::GsfElectronCollection> inputElectronsToken_;
  edm::EDGetTokenT<reco::PhotonCollection> inputPhotonsToken_;
  // new -> old map
  edm::EDGetTokenT<ElectronRefMap> electronMapToken_;
  edm::EDGetTokenT<PhotonRefMap> photonMapToken_;

  std::string electronsMapName_;
  std::string photonsMapName_;
};

#endif
