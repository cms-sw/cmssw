//
// $Id: PATGenericParticleProducer.cc,v 1.1.2.1 2008/05/28 13:46:01 gpetrucc Exp $
//

#include "PhysicsTools/PatAlgos/plugins/PATGenericParticleProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/View.h"
#include <memory>

using namespace pat;

PATGenericParticleProducer::PATGenericParticleProducer(const edm::ParameterSet & iConfig) :
  isolator_(iConfig.exists("isolation") ? iConfig.getParameter<edm::ParameterSet>("isolation") : edm::ParameterSet(), false) 
{
  // initialize the configurables
  src_ = iConfig.getParameter<edm::InputTag>( "src" );

  // RECO embedding
  embedTrack_        = iConfig.getParameter<bool>( "embedTrack" );
  embedGsfTrack_     = iConfig.getParameter<bool>( "embedGsfTrack" );
  embedStandalone_   = iConfig.getParameter<bool>( "embedStandAloneMuon" );
  embedCombined_     = iConfig.getParameter<bool>( "embedCombinedMuon" );
  embedSuperCluster_ = iConfig.getParameter<bool>( "embedSuperCluster" );
  embedTracks_       = iConfig.getParameter<bool>( "embedMultipleTracks" );
  embedCaloTower_    = iConfig.getParameter<bool>( "embedCaloTower" );
  
  // MC matching configurables
  addGenMatch_   = iConfig.getParameter<bool>( "addGenMatch" );
  genMatchSrc_    = iConfig.getParameter<edm::InputTag>( "genParticleMatch" );

  // Trigger matching configurables
  addTrigMatch_  = iConfig.getParameter<bool>( "addTrigMatch" );
  trigPrimSrc_   = iConfig.getParameter<std::vector<edm::InputTag> >( "trigPrimMatch" );

  // quality
  addQuality_ = iConfig.getParameter<bool>("addQuality");
  qualitySrc_ = iConfig.getParameter<edm::InputTag>("qualitySource");

  // produces vector of particles
  produces<std::vector<GenericParticle> >();

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

PATGenericParticleProducer::~PATGenericParticleProducer() {
}

void PATGenericParticleProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
  // Get the vector of GenericParticle's from the event
  edm::Handle<edm::View<reco::Candidate> > cands;
  iEvent.getByLabel(src_, cands);

  // prepare isolation
  if (isolator_.enabled()) isolator_.beginEvent(iEvent);

  // prepare IsoDeposits
  std::vector<edm::Handle<edm::ValueMap<IsoDeposit> > > deposits(isoDepositLabels_.size());
  for (size_t j = 0, nd = deposits.size(); j < nd; ++j) {
    iEvent.getByLabel(isoDepositLabels_[j].second, deposits[j]);
  }

  // prepare the MC matching
  edm::Handle<edm::Association<reco::GenParticleCollection> > genMatch;
  if (addGenMatch_) iEvent.getByLabel(genMatchSrc_, genMatch);

  // prepare the quality
  edm::Handle<edm::ValueMap<float> > qualities;
  if (addQuality_) iEvent.getByLabel(qualitySrc_, qualities);

  // loop over cands
  std::vector<GenericParticle> * PATGenericParticles = new std::vector<GenericParticle>(); 
  for (edm::View<reco::Candidate>::const_iterator itGenericParticle = cands->begin(); itGenericParticle != cands->end(); itGenericParticle++) {
    // construct the GenericParticle from the ref -> save ref to original object
    unsigned int idx = itGenericParticle - cands->begin();
    edm::RefToBase<reco::Candidate> candRef = cands->refAt(idx);

    PATGenericParticles->push_back(GenericParticle(candRef));
    GenericParticle & aGenericParticle = PATGenericParticles->back();

    // embed RECO
    if (embedTrack_)        aGenericParticle.embedTrack();
    if (embedGsfTrack_)     aGenericParticle.embedGsfTrack();
    if (embedTracks_)       aGenericParticle.embedTracks();
    if (embedStandalone_)   aGenericParticle.embedStandalone();
    if (embedCombined_)     aGenericParticle.embedCombined();
    if (embedSuperCluster_) aGenericParticle.embedSuperCluster();
    if (embedCaloTower_)    aGenericParticle.embedCaloTower();

    // matches to fired trigger primitives
    if ( addTrigMatch_ ) {
      for ( size_t i = 0; i < trigPrimSrc_.size(); ++i ) {
        edm::Handle<edm::Association<TriggerPrimitiveCollection> > trigMatch;
        iEvent.getByLabel(trigPrimSrc_[i], trigMatch);
        TriggerPrimitiveRef trigPrim = (*trigMatch)[candRef];
        if ( trigPrim.isNonnull() && trigPrim.isAvailable() ) {
          aGenericParticle.addTriggerMatch(*trigPrim);
        }
      }
    }

    // isolation
    if (isolator_.enabled()) {
        isolator_.fill(*cands, idx, isolatorTmpStorage_);
        typedef pat::helper::MultiIsolator::IsolationValuePairs IsolationValuePairs;
        // better to loop backwards, so the vector is resized less times
        for (IsolationValuePairs::const_reverse_iterator it = isolatorTmpStorage_.rbegin(), ed = isolatorTmpStorage_.rend(); it != ed; ++it) {
            aGenericParticle.setIsolation(it->first, it->second);
        }
    }

    // isodeposit
    for (size_t j = 0, nd = deposits.size(); j < nd; ++j) {
        aGenericParticle.setIsoDeposit(isoDepositLabels_[j].first, (*deposits[j])[candRef]);
    }

    // match to generated final state particle
    if (addGenMatch_) {
      reco::GenParticleRef genGenericParticle = (*genMatch)[candRef];
      if (genGenericParticle.isNonnull() && genGenericParticle.isAvailable() ) {
        aGenericParticle.setGenParticle(*genGenericParticle);
      } // leave empty if no match found
    }

    if (addQuality_) {
      aGenericParticle.setQuality( (*qualities)[candRef] );
    }

    // add the GenericParticle to the vector of GenericParticles
    PATGenericParticles->push_back(aGenericParticle);
  }

  // sort GenericParticles in ET
  std::sort(PATGenericParticles->begin(), PATGenericParticles->end(), eTComparator_);

  // put genEvt object in Event
  std::auto_ptr<std::vector<GenericParticle> > myGenericParticles(PATGenericParticles);
  iEvent.put(myGenericParticles);
  if (isolator_.enabled()) isolator_.endEvent();

}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATGenericParticleProducer);
