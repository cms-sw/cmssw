#include "RecoEgamma/EgammaElectronProducers/plugins/LowPtGsfElectronProducer.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/GsfElectronAlgo.h"
#include <iostream>

using namespace reco;

LowPtGsfElectronProducer::LowPtGsfElectronProducer( const edm::ParameterSet& cfg, 
						    const gsfAlgoHelpers::HeavyObjectCache* hoc )
  : GsfElectronBaseProducer(cfg,hoc)
{}

LowPtGsfElectronProducer::~LowPtGsfElectronProducer()
{}

void LowPtGsfElectronProducer::produce( edm::Event& event, const edm::EventSetup& setup )
{
  std::cout << "[LowPtGsfElectronProducer::produce]" << std::endl; // @@
  auto electrons = std::make_unique<GsfElectronCollection>();
  edm::Handle<reco::GsfElectronCoreCollection> coreElectrons;
  event.getByToken(inputCfg_.gsfElectronCores,coreElectrons);
  std::cout << "[LowPtGsfElectronProducer::produce]" << coreElectrons->size() << std::endl; //@@
  for ( unsigned int ii=0; ii < coreElectrons->size(); ++ii ) {
    const GsfElectronCoreRef ref = edm::Ref<GsfElectronCoreCollection>(coreElectrons,ii);
    GsfElectron* ele = new GsfElectron(ref);
    const GsfTrackRef& gsf = ref->gsfTrack();
    ele->setP4(GsfElectron::P4_FROM_SUPER_CLUSTER,Candidate::LorentzVector(gsf->px(),gsf->py(),gsf->pz(),0.511E-3),0,true) ;
    LogTrace("GsfElectronAlgo")<<"Constructed new electron with energy  "<< ele->p4().e() ;
    electrons->push_back(*ele) ;
  }
  std::cout << "[LowPtGsfElectronCoreProducer::produce]" << electrons->size() << std::endl; //@@
  event.put(std::move(electrons));
//  beginEvent(event,setup);
//  algo_->completeElectrons(globalCache());
//  fillEvent(event);
//  endEvent();
}
