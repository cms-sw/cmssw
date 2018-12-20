#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCoreFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoEgamma/EgammaElectronProducers/plugins/LowPtGsfElectronCoreProducer.h"

LowPtGsfElectronCoreProducer::LowPtGsfElectronCoreProducer( const edm::ParameterSet& config )
  : GsfElectronCoreBaseProducer(config)
{
  superClusters_ = consumes<reco::SuperClusterCollection>(config.getParameter<edm::InputTag>("superClusters"));
  superClusterRefs_ = consumes< edm::ValueMap<reco::SuperClusterRef> >(config.getParameter<edm::InputTag>("superClusters"));
}

LowPtGsfElectronCoreProducer::~LowPtGsfElectronCoreProducer() {}

void LowPtGsfElectronCoreProducer::produce( edm::Event& event, const edm::EventSetup& setup ) {

  // Output collection
  auto electrons = std::make_unique<reco::GsfElectronCoreCollection>();

  // Init
  GsfElectronCoreBaseProducer::initEvent(event,setup) ;
  if ( !useGsfPfRecTracks_ ) { edm::LogError("useGsfPfRecTracks_ is (redundantly) set to False!"); }
  if ( !gsfPfRecTracksH_.isValid() ) { edm::LogError("gsfPfRecTracks handle is invalid!"); }
  if ( !gsfTracksH_.isValid() ) { edm::LogError("gsfTracks handle is invalid!"); }

  edm::Handle<reco::SuperClusterCollection> superClusters;
  event.getByToken(superClusters_,superClusters);
  if ( !superClusters.isValid() ) { edm::LogError("Problem with superClusters handle"); }

  edm::Handle< edm::ValueMap<reco::SuperClusterRef> > superClusterRefs;
  event.getByToken(superClusterRefs_,superClusterRefs);
  if ( !superClusterRefs.isValid() ) { edm::LogError("Problem with superClusterRefs handle"); }

  // Create ElectronCore objects
  for ( size_t ipfgsf = 0; ipfgsf < gsfPfRecTracksH_->size(); ++ipfgsf ) {

    // Refs to GSF(PF) objects and SC
    reco::GsfPFRecTrackRef pfgsf(gsfPfRecTracksH_, ipfgsf);
    reco::GsfTrackRef gsf = pfgsf->gsfTrackRef();
    const reco::SuperClusterRef sc = (*superClusterRefs)[pfgsf];

    // Construct GsfElectronCore and add GSF(PF) track and SC
    reco::GsfElectronCore* core = new reco::GsfElectronCore(gsf);
    if ( core->ecalDrivenSeed() ) { delete core; return; }

    // Add GSF(PF) track information
    GsfElectronCoreBaseProducer::fillElectronCore(core);

    // Add super cluster
    core->setSuperCluster(sc);

    // Store
    electrons->push_back(*core);

    // Delete object
    delete core;

  }

  event.put(std::move(electrons));

}

//////////////////////////////////////////////////////////////////////////////////////////
//
void LowPtGsfElectronCoreProducer::fillDescriptions( edm::ConfigurationDescriptions& descriptions )
{
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("gsfPfRecTracks",edm::InputTag("lowPtGsfElePfGsfTracks")) ;
  desc.add<edm::InputTag>("gsfTracks",edm::InputTag("lowPtGsfEleGsfTracks")) ;
  desc.add<edm::InputTag>("ctfTracks",edm::InputTag("generalTracks")) ;
  desc.add<edm::InputTag>("superClusters",edm::InputTag("lowPtGsfElectronSuperClusters")) ;
  desc.add<bool>("useGsfPfRecTracks",true) ;
  descriptions.add("lowPtGsfElectronCores",desc);
}
