#include "RecoEgamma/EgammaElectronProducers/plugins/LowPtGsfElectronCoreProducer.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include <map>

using namespace reco ;

LowPtGsfElectronCoreProducer::LowPtGsfElectronCoreProducer( const edm::ParameterSet& config )
  : GsfElectronCoreBaseProducer(config)
{
  superClusters_ = consumes<reco::SuperClusterCollection>(config.getParameter<edm::InputTag>("superClusters"));
  superClusterRefs_ = consumes< edm::ValueMap<reco::SuperClusterRef> >(config.getParameter<edm::InputTag>("superClusters"));
}

void LowPtGsfElectronCoreProducer::produce( edm::Event& event, const edm::EventSetup& setup ) {

  // Output collection
  auto electrons = std::make_unique<GsfElectronCoreCollection>();

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

    // Construct and keep ElectronCore if GSF(PF) track and SC are present
    GsfElectronCore* core = new GsfElectronCore(gsf);
    if ( core->ecalDrivenSeed() ) { delete core; return; }

    // Add GSF(PF) track information
    GsfElectronCoreBaseProducer::fillElectronCore(core);

    // Add super cluster
    core->setSuperCluster(sc);

    // Store
    electrons->push_back(*core);

  }

  //std::cout << "[LowPtGsfElectronCoreProducer::produce] " << electrons->size() << std::endl; //@@
  event.put(std::move(electrons));

}

void LowPtGsfElectronCoreProducer::produceTrackerDrivenCore( const GsfTrackRef& gsfTrackRef,
							     GsfElectronCoreCollection* electrons ) {
  //@@ This method not currently used!
  GsfElectronCore* eleCore = new GsfElectronCore(gsfTrackRef);
  if (eleCore->ecalDrivenSeed()) { delete eleCore; return; }
  GsfElectronCoreBaseProducer::fillElectronCore(eleCore);
  electrons->push_back(*eleCore);
}

LowPtGsfElectronCoreProducer::~LowPtGsfElectronCoreProducer() {}

