#ifndef CalibratedPhotonProducer_h
#define CalibratedPhotonProducer_h

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "EgammaAnalysis/ElectronTools/interface/PhotonEnergyCalibratorRun2.h"

#include <vector>
#include <random>
#include <TRandom2.h>

template<typename T>
class CalibratedPhotonProducerRun2T: public edm::stream::EDProducer<> {
public:
  explicit CalibratedPhotonProducerRun2T( const edm::ParameterSet & ) ;
  ~CalibratedPhotonProducerRun2T() override;
  void produce( edm::Event &, const edm::EventSetup & ) override ;

private:
  edm::EDGetTokenT<edm::View<T> > thePhotonToken;
  PhotonEnergyCalibratorRun2 theEnCorrectorRun2;
  std::unique_ptr<TRandom> theSemiDeterministicRng;
};

template<typename T>
CalibratedPhotonProducerRun2T<T>::CalibratedPhotonProducerRun2T( const edm::ParameterSet & conf ) :
  thePhotonToken(consumes<edm::View<T> >(conf.getParameter<edm::InputTag>("photons"))),
  theEnCorrectorRun2(conf.getParameter<bool>("isMC"), conf.getParameter<bool>("isSynchronization"), conf.getParameter<std::string >("correctionFile")) {

  if (conf.existsAs<bool>("semiDeterministic") && conf.getParameter<bool>("semiDeterministic")) {
    theSemiDeterministicRng.reset(new TRandom2());
    theEnCorrectorRun2.initPrivateRng(theSemiDeterministicRng.get());
  }
  produces<std::vector<T> >();
}

template<typename T>
CalibratedPhotonProducerRun2T<T>::~CalibratedPhotonProducerRun2T()
{}

template<typename T>
void
CalibratedPhotonProducerRun2T<T>::produce( edm::Event & iEvent, const edm::EventSetup & iSetup ) {

  edm::Handle<edm::View<T> > in;
  iEvent.getByToken(thePhotonToken, in);

  if (theSemiDeterministicRng && !in->empty()) { // no need to set a seed if in is empty
      const auto & first = in->front();
      std::seed_seq seeder = {int(iEvent.id().event()), int(iEvent.id().luminosityBlock()), int(iEvent.id().run()),
          int(in->size()), int(std::numeric_limits<int>::max()*first.phi()/M_PI) & 0xFFF, int(first.pdgId())};
      uint32_t seed = 0, tries = 10;
      do {
          seeder.generate(&seed,&seed+1); tries++;
      } while (seed == 0 && tries < 10);
      theSemiDeterministicRng->SetSeed(seed ? seed : iEvent.id().event());
  }

  std::unique_ptr<std::vector<T> > out(new std::vector<T>());
  out->reserve(in->size());   
  
  for (const T &ele : *in) {
    out->push_back(ele);
    theEnCorrectorRun2.calibrate(out->back(), iEvent.id().run(), iEvent.streamID());
  }
    
  iEvent.put(std::move(out));
}

typedef CalibratedPhotonProducerRun2T<reco::Photon> CalibratedPhotonProducerRun2;
typedef CalibratedPhotonProducerRun2T<pat::Photon> CalibratedPatPhotonProducerRun2;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(CalibratedPhotonProducerRun2);
DEFINE_FWK_MODULE(CalibratedPatPhotonProducerRun2);

#endif
