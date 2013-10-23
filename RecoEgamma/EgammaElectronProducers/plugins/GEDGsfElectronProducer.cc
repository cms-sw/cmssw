
#include "GEDGsfElectronProducer.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/GsfElectronAlgo.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "RecoParticleFlow/PFProducer/interface/GsfElectronEqual.h"

#include <iostream>
#include <string>

using namespace reco;

GEDGsfElectronProducer::GEDGsfElectronProducer( const edm::ParameterSet & cfg )
 : GsfElectronBaseProducer(cfg)
 {
   egmPFCandidateCollection_ = consumes<reco::PFCandidateCollection>(cfg.getParameter<edm::InputTag>("egmPFCandidatesTag"));
   outputValueMapLabel_ = cfg.getParameter<std::string>("outputEGMPFValueMap");

   produces<edm::ValueMap<reco::GsfElectronRef> >(outputValueMapLabel_);
}

GEDGsfElectronProducer::~GEDGsfElectronProducer()
 {}

// ------------ method called to produce the data  ------------
void GEDGsfElectronProducer::produce( edm::Event & event, const edm::EventSetup & setup )
 {
  beginEvent(event,setup) ;
  algo_->completeElectrons() ;
  fillEvent(event) ;

  // ValueMap
  std::auto_ptr<edm::ValueMap<reco::GsfElectronRef> > valMap_p(new edm::ValueMap<reco::GsfElectronRef>);
  edm::ValueMap<reco::GsfElectronRef>::Filler valMapFiller(*valMap_p);
  matchWithPFCandidates(event,valMapFiller);
  valMapFiller.fill();
  event.put(valMap_p,outputValueMapLabel_);  
  // Done with the ValueMap

  endEvent() ;
 }

void GEDGsfElectronProducer::matchWithPFCandidates(edm::Event & event, edm::ValueMap<reco::GsfElectronRef>::Filler & filler)
{
  // Read the collection of PFCandidates
  edm::Handle<reco::PFCandidateCollection> pfCandidates;
  
  bool found = event.getByToken(egmPFCandidateCollection_, pfCandidates);
  if(!found) {
    edm::LogError("GEDGsfElectronProducer")
       <<" cannot get PFCandidates! ";
  }

  //Loop over the collection of PFFCandidates
  reco::PFCandidateCollection::const_iterator it = pfCandidates->begin();
  reco::PFCandidateCollection::const_iterator itend = pfCandidates->end() ;
  std::vector<reco::GsfElectronRef> values;

  for ( ; it != itend ; ++it) {
    reco::GsfElectronRef myRef;
    // First check that the GsfTrack is non null
    if( it->gsfTrackRef().isNonnull()) {
      // now look for the corresponding GsfElectron
      GsfElectronEqual myEqual(it->gsfTrackRef());
      const reco::GsfElectronCollection::const_iterator itcheck=
	std::find_if(orphanHandle()->begin(),orphanHandle()->end(),myEqual);
      if (itcheck != orphanHandle()->end()) {
	// Build the Ref from the handle and the index
	myRef = reco::GsfElectronRef(orphanHandle(),itcheck-orphanHandle()->begin());
      }
    }
    values.push_back(myRef);
  }
  filler.insert(pfCandidates,values.begin(),values.end());
}

