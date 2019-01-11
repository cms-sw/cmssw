#include "GsfElectronEcalDrivenProducer.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/GsfElectronAlgo.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"


#include <iostream>

using namespace reco;

GsfElectronEcalDrivenProducer::GsfElectronEcalDrivenProducer( const edm::ParameterSet & cfg, const gsfAlgoHelpers::HeavyObjectCache* hoc )
  : GsfElectronBaseProducer(cfg,hoc)
 {}

GsfElectronEcalDrivenProducer::~GsfElectronEcalDrivenProducer()
 {}

// ------------ method called to produce the data  ------------
void GsfElectronEcalDrivenProducer::produce( edm::Event & event, const edm::EventSetup & setup )
 {
  beginEvent(event,setup) ;
  algo_->completeElectrons(globalCache()) ;
  fillEvent(event) ;
  endEvent() ;
 }
