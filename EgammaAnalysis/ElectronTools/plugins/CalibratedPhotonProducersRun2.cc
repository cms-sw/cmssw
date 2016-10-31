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

template<typename T>
class CalibratedPhotonProducerRun2T: public edm::stream::EDProducer<> {
public:
  explicit CalibratedPhotonProducerRun2T( const edm::ParameterSet & ) ;
  virtual ~CalibratedPhotonProducerRun2T();
  virtual void produce( edm::Event &, const edm::EventSetup & ) override ;

private:
  edm::EDGetTokenT<edm::View<T> > thePhotonToken;
  PhotonEnergyCalibratorRun2 theEnCorrectorRun2;
};

template<typename T>
CalibratedPhotonProducerRun2T<T>::CalibratedPhotonProducerRun2T( const edm::ParameterSet & conf ) :
  thePhotonToken(consumes<edm::View<T> >(conf.getParameter<edm::InputTag>("photons"))),
  theEnCorrectorRun2(conf.getParameter<bool>("isMC"), conf.getParameter<bool>("isSynchronization"), conf.getParameter<std::string >("correctionFile")) {

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

  std::auto_ptr<std::vector<T> > out(new std::vector<T>());
  out->reserve(in->size());   
  
  for (const T &ele : *in) {
    out->push_back(ele);
    theEnCorrectorRun2.calibrate(out->back(), iEvent.id().run(), iEvent.streamID());
  }
    
  iEvent.put(out);
}

typedef CalibratedPhotonProducerRun2T<reco::Photon> CalibratedPhotonProducerRun2;
typedef CalibratedPhotonProducerRun2T<pat::Photon> CalibratedPatPhotonProducerRun2;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(CalibratedPhotonProducerRun2);
DEFINE_FWK_MODULE(CalibratedPatPhotonProducerRun2);

#endif
