
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

/* void GsfElectronEcalDrivenProducer::fillDescriptions( edm::ConfigurationDescriptions & descriptions )
 {
  edm::ParameterSetDescription desc ;
  GsfElectronBaseProducer::fillDescription(desc) ;

  // input collections
  desc.add<edm::InputTag>("ecalDrivenGsfElectronCoresTag",edm::InputTag("ecalDrivenGsfElectronCores")) ;

  // preselection parameters (ecal driven electrons)
  desc.add<double>("minSCEtBarrel",4.0) ;
  desc.add<double>("minSCEtEndcaps",4.0) ;
  desc.add<double>("minEOverPBarrel",0.0) ;
  desc.add<double>("maxEOverPBarrel",999999999.) ;
  desc.add<double>("minEOverPEndcaps",0.0) ;
  desc.add<double>("maxEOverPEndcaps",999999999.) ;
  desc.add<double>("maxDeltaEtaBarrel",0.02) ;
  desc.add<double>("maxDeltaEtaEndcaps",0.02) ;
  desc.add<double>("maxDeltaPhiBarrel",0.15) ;
  desc.add<double>("maxDeltaPhiEndcaps",0.15) ;
  desc.add<double>("hOverEConeSize",0.15) ;
  desc.add<double>("hOverEPtMin",0.) ;
  desc.add<double>("maxHOverEBarrel",0.15) ;
  desc.add<double>("maxHOverEEndcaps",0.15) ;
  desc.add<double>("maxHBarrel",0.0) ;
  desc.add<double>("maxHEndcaps",0.0) ;
  desc.add<double>("maxSigmaIetaIetaBarrel",999999999.) ;
  desc.add<double>("maxSigmaIetaIetaEndcaps",999999999.) ;
  desc.add<double>("maxFbremBarrel",999999999.) ;
  desc.add<double>("maxFbremEndcaps",999999999.) ;
  desc.add<bool>("isBarrel",false) ;
  desc.add<bool>("isEndcaps",false) ;
  desc.add<bool>("isFiducial",false) ;
  desc.add<bool>("seedFromTEC",true) ;
  desc.add<double>("maxTIP",999999999.) ;
  desc.add<double>("minMVA",-0.4) ;

  descriptions.add("produceEcalDrivenGsfElectrons",desc) ;
 }
 */
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


