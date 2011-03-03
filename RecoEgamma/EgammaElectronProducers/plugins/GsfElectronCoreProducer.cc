
#include "GsfElectronCoreProducer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronCoreFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include <map>

using namespace reco ;

// void GsfElectronCoreProducer::fillDescriptions( edm::ConfigurationDescriptions & descriptions )
//  {
//   edm::ParameterSetDescription desc ;
//   GsfElectronCoreBaseProducer::fillDescriptions(desc) ;
//   desc.add<edm::InputTag>("ecalDrivenGsfElectronCoresTag",edm::InputTag("ecalDrivenGsfElectronCores")) ;
//   desc.add<edm::InputTag>("pfSuperClusters",edm::InputTag("pfElectronTranslator:pf")) ;
//   desc.add<edm::InputTag>("pfSuperClusterTrackMap",edm::InputTag("pfElectronTranslator:pf")) ;
//   descriptions.add("produceGsfElectronCores",desc) ;
//  }

GsfElectronCoreProducer::GsfElectronCoreProducer( const edm::ParameterSet & config )
 : GsfElectronCoreBaseProducer(config)
 {
  edCoresTag_ = config.getParameter<edm::InputTag>("ecalDrivenGsfElectronCoresTag") ;
  pfSuperClustersTag_ = config.getParameter<edm::InputTag>("pfSuperClusters") ;
  pfSuperClusterTrackMapTag_ = config.getParameter<edm::InputTag>("pfSuperClusterTrackMap") ;
 }

void GsfElectronCoreProducer::produce( edm::Event & event, const edm::EventSetup & setup )
 {
  // base input
  GsfElectronCoreBaseProducer::initEvent(event,setup) ;

  // output
  std::auto_ptr<GsfElectronCoreCollection> electrons(new GsfElectronCoreCollection) ;

  // event input
  event.getByLabel(edCoresTag_,edCoresH_) ;
  event.getByLabel(pfSuperClustersTag_,pfClustersH_) ;
  event.getByLabel(pfSuperClusterTrackMapTag_,pfClusterTracksH_) ;

  // loop on pure tracker driven tracks
  if (useGsfPfRecTracks_)
   {
    const GsfPFRecTrackCollection * gsfPfRecTrackCollection = gsfPfRecTracksH_.product() ;
    GsfPFRecTrackCollection::const_iterator gsfPfRecTrack ;
    for ( gsfPfRecTrack=gsfPfRecTrackCollection->begin() ;
          gsfPfRecTrack!=gsfPfRecTrackCollection->end() ;
          ++gsfPfRecTrack )
     {
      const GsfTrackRef gsfTrackRef = gsfPfRecTrack->gsfTrackRef() ;
      produceTrackerDrivenCore(gsfTrackRef,electrons.get()) ;
     }
   }
  else
   {
    const GsfTrackCollection * gsfTrackCollection = gsfTracksH_.product() ;
    for ( unsigned int i=0 ; i<gsfTrackCollection->size() ; ++i )
     {
      const GsfTrackRef gsfTrackRef = edm::Ref<GsfTrackCollection>(gsfTracksH_,i) ;
      produceTrackerDrivenCore(gsfTrackRef,electrons.get()) ;
     }
   }

  // clone and complete ecal driven electrons
  const GsfElectronCoreCollection * edCoresCollection = edCoresH_.product() ;
  GsfElectronCoreCollection::const_iterator edCoreIter ;
  for
   ( edCoreIter = edCoresCollection->begin() ;
     edCoreIter != edCoresCollection->end() ;
     edCoreIter++ )
   {
    GsfElectronCore * eleCore = edCoreIter->clone() ;
    const GsfTrackRef & gsfTrackRef = eleCore->gsfTrack() ;
    eleCore->setPflowSuperCluster((*pfClusterTracksH_)[gsfTrackRef]) ;
    electrons->push_back(*eleCore) ;
    delete eleCore ;
   }

  event.put(electrons) ;
 }

void GsfElectronCoreProducer::produceTrackerDrivenCore( const GsfTrackRef & gsfTrackRef, GsfElectronCoreCollection * electrons )
 {
  GsfElectronCore * eleCore = new GsfElectronCore(gsfTrackRef) ;
  if (eleCore->ecalDrivenSeed())
   { delete eleCore ; return ; }

  GsfElectronCoreBaseProducer::fillElectronCore(eleCore) ;

  eleCore->setPflowSuperCluster((*pfClusterTracksH_)[gsfTrackRef]) ;
  if (eleCore->superCluster().isNull())
   { LogDebug("GsfElectronCoreProducer")<<"GsfTrack with no associated SuperCluster at all." ; }
  else
   {
    electrons->push_back(*eleCore) ;
   }
  delete eleCore ;
 }

GsfElectronCoreProducer::~GsfElectronCoreProducer()
 {}

