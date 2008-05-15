//
// $Id: PATPhotonProducer.cc,v 1.2 2008/01/26 20:20:34 gpetrucc Exp $
//

#include "PhysicsTools/PatAlgos/plugins/PATPhotonProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/View.h"
#include <memory>

using namespace pat;

PATPhotonProducer::PATPhotonProducer(const edm::ParameterSet & iConfig) {
  // initialize the configurables
  photonSrc_         = iConfig.getParameter<edm::InputTag>("photonSource");
  
  // produces vector of photons
  produces<std::vector<Photon> >();
}

PATPhotonProducer::~PATPhotonProducer() {
}

void PATPhotonProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
 
  // Get the vector of Photon's from the event
  edm::Handle<edm::View<PhotonType> > photons;
  iEvent.getByLabel(photonSrc_, photons);

  // loop over photons
  std::vector<Photon> * PATPhotons = new std::vector<Photon>(); 
  for (edm::View<PhotonType>::const_iterator itPhoton = photons->begin(); itPhoton != photons->end(); itPhoton++) {
    // construct the Photon from the ref -> save ref to original object
    unsigned int idx = itPhoton - photons->begin();
    edm::RefToBase<PhotonType> photonRef = photons->refAt(idx);
    Photon aPhoton(photonRef);

    // here comes the extra functionality

    // add the Photon to the vector of Photons
    PATPhotons->push_back(aPhoton);
  }

  // sort Photons in ET
  std::sort(PATPhotons->begin(), PATPhotons->end(), eTComparator_);

  // put genEvt object in Event
  std::auto_ptr<std::vector<Photon> > myPhotons(PATPhotons);
  iEvent.put(myPhotons);

}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATPhotonProducer);
