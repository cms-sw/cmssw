#include "RecoEgamma/EgammaElectronProducers/plugins/LowPtGsfElectronCoreProducer.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
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
{}

void LowPtGsfElectronCoreProducer::produce( edm::Event& event, const edm::EventSetup& setup ) {
  GsfElectronCoreBaseProducer::initEvent(event,setup) ;
  auto electrons = std::make_unique<GsfElectronCoreCollection>();
  if (useGsfPfRecTracks_) {
    const GsfPFRecTrackCollection * gsfPfRecTrackCollection = gsfPfRecTracksH_.product() ;
    GsfPFRecTrackCollection::const_iterator gsfPfRecTrack ;
    for ( gsfPfRecTrack=gsfPfRecTrackCollection->begin() ;
	  gsfPfRecTrack!=gsfPfRecTrackCollection->end() ;
	  ++gsfPfRecTrack ) {
      const GsfTrackRef gsfTrackRef = gsfPfRecTrack->gsfTrackRef() ;
      produceTrackerDrivenCore(gsfTrackRef,electrons.get()) ;
    }
  } else {
    const GsfTrackCollection * gsfTrackCollection = gsfTracksH_.product() ;
    for ( unsigned int i=0 ; i<gsfTrackCollection->size() ; ++i ) {
      const GsfTrackRef gsfTrackRef = edm::Ref<GsfTrackCollection>(gsfTracksH_,i) ;
      produceTrackerDrivenCore(gsfTrackRef,electrons.get()) ;
    }
  }
  std::cout << "[LowPtGsfElectronCoreProducer::produce]" << electrons->size() << std::endl; //@@
  event.put(std::move(electrons));
}

void LowPtGsfElectronCoreProducer::produceTrackerDrivenCore( const GsfTrackRef& gsfTrackRef, 
							     GsfElectronCoreCollection* electrons ) {
  GsfElectronCore* eleCore = new GsfElectronCore(gsfTrackRef) ;
  if (eleCore->ecalDrivenSeed()) { delete eleCore ; return ; }
  GsfElectronCoreBaseProducer::fillElectronCore(eleCore) ;
  electrons->push_back(*eleCore) ;
}

LowPtGsfElectronCoreProducer::~LowPtGsfElectronCoreProducer() {}

