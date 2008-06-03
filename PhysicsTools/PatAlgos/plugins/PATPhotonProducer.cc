//
// $Id: PATPhotonProducer.cc,v 1.5.2.1 2008/05/31 19:34:14 lowette Exp $
//

#include "PhysicsTools/PatAlgos/plugins/PATPhotonProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include <memory>

using namespace pat;

PATPhotonProducer::PATPhotonProducer(const edm::ParameterSet & iConfig) :
  isolator_(iConfig.exists("isolation") ? iConfig.getParameter<edm::ParameterSet>("isolation") : edm::ParameterSet(), false) 
{
  // initialize the configurables
  photonSrc_         = iConfig.getParameter<edm::InputTag>("photonSource");
  embedSuperCluster_ = iConfig.getParameter<bool>         ("embedSuperCluster");

   // MC matching configurables
  addGenMatch_       = iConfig.getParameter<bool>         ( "addGenMatch" );
  genMatchSrc_       = iConfig.getParameter<edm::InputTag>( "genParticleMatch" );
  
  // produces vector of photons
  produces<std::vector<Photon> >();

  if (iConfig.exists("isoDeposits")) {
     edm::ParameterSet depconf = iConfig.getParameter<edm::ParameterSet>("isoDeposits");
     if (depconf.exists("tracker")) isoDepositLabels_.push_back(std::make_pair(TrackerIso, depconf.getParameter<edm::InputTag>("tracker")));
     if (depconf.exists("ecal"))    isoDepositLabels_.push_back(std::make_pair(ECalIso, depconf.getParameter<edm::InputTag>("ecal")));
     if (depconf.exists("hcal"))    isoDepositLabels_.push_back(std::make_pair(HCalIso, depconf.getParameter<edm::InputTag>("hcal")));
     if (depconf.exists("user")) {
        std::vector<edm::InputTag> userdeps = depconf.getParameter<std::vector<edm::InputTag> >("user");
        std::vector<edm::InputTag>::const_iterator it = userdeps.begin(), ed = userdeps.end();
        int key = UserBaseIso;
        for ( ; it != ed; ++it, ++key) {
            isoDepositLabels_.push_back(std::make_pair(IsolationKeys(key), *it));
        }
     }
  }
}

PATPhotonProducer::~PATPhotonProducer() {
}

void PATPhotonProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
 
  // Get the vector of Photon's from the event
  edm::Handle<edm::View<PhotonType> > photons;
  iEvent.getByLabel(photonSrc_, photons);

  // prepare the MC matching
  edm::Handle<edm::Association<reco::GenParticleCollection> > genMatch;
  if (addGenMatch_) {
    iEvent.getByLabel(genMatchSrc_, genMatch);
  }

  if (isolator_.enabled()) isolator_.beginEvent(iEvent);

  std::vector<edm::Handle<edm::ValueMap<IsoDeposit> > > deposits(isoDepositLabels_.size());
  for (size_t j = 0, nd = deposits.size(); j < nd; ++j) {
    iEvent.getByLabel(isoDepositLabels_[j].second, deposits[j]);
  }

  // loop over photons
  std::vector<Photon> * PATPhotons = new std::vector<Photon>(); 
  for (edm::View<PhotonType>::const_iterator itPhoton = photons->begin(); itPhoton != photons->end(); itPhoton++) {
    // construct the Photon from the ref -> save ref to original object
    unsigned int idx = itPhoton - photons->begin();
    edm::RefToBase<PhotonType> photonRef = photons->refAt(idx);
    Photon aPhoton(photonRef);
    if (embedSuperCluster_) aPhoton.embedSuperCluster();

    // store the match to the generated final state photons
    if (addGenMatch_) {
      reco::GenParticleRef genPhoton = (*genMatch)[photonRef];
      if (genPhoton.isNonnull() && genPhoton.isAvailable() ) {
        aPhoton.setGenPhoton(*genPhoton);
      } // leave empty if no match found
    }

    // here comes the extra functionality
    if (isolator_.enabled()) {
        isolator_.fill(*photons, idx, isolatorTmpStorage_);
        typedef pat::helper::MultiIsolator::IsolationValuePairs IsolationValuePairs;
        // better to loop backwards, so the vector is resized less times
        for (IsolationValuePairs::const_reverse_iterator it = isolatorTmpStorage_.rbegin(), ed = isolatorTmpStorage_.rend(); it != ed; ++it) {
            aPhoton.setIsolation(it->first, it->second);
        }
    }

    for (size_t j = 0, nd = deposits.size(); j < nd; ++j) {
        aPhoton.setIsoDeposit(isoDepositLabels_[j].first, (*deposits[j])[photonRef]);
    }

    // add the Photon to the vector of Photons
    PATPhotons->push_back(aPhoton);
  }

  // sort Photons in ET
  std::sort(PATPhotons->begin(), PATPhotons->end(), eTComparator_);

  // put genEvt object in Event
  std::auto_ptr<std::vector<Photon> > myPhotons(PATPhotons);
  iEvent.put(myPhotons);
  if (isolator_.enabled()) isolator_.endEvent();

}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATPhotonProducer);
