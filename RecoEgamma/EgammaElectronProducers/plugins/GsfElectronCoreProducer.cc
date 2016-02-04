
#include "GsfElectronCoreProducer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronCoreFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
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

  // additional input
  edm::Handle<GsfElectronCoreCollection> edCoresH ;
  event.getByLabel(edCoresTag_,edCoresH) ;
  edm::Handle<SuperClusterCollection> pfClustersH ;
  event.getByLabel(pfSuperClustersTag_,pfClustersH) ;
  edm::Handle<edm::ValueMap<reco::SuperClusterRef> > pfClusterTracksH ;
  event.getByLabel(pfSuperClusterTrackMapTag_,pfClusterTracksH) ;

  // loop on pure tracker driven tracks
  const GsfTrackCollection * gsfTrackCollection = gsfTracksH_.product() ;
  for ( unsigned int i=0 ; i<gsfTrackCollection->size() ; ++i )
   {
    const GsfTrackRef gsfTrackRef = edm::Ref<GsfTrackCollection>(gsfTracksH_,i) ;
    GsfElectronCore * ele = new GsfElectronCore(gsfTrackRef) ;

    if (ele->ecalDrivenSeed())
     { delete ele ; continue ; }

    GsfElectronCoreBaseProducer::fillElectronCore(ele) ;

    ele->setPflowSuperCluster((*pfClusterTracksH)[gsfTrackRef]) ;
    if (!(ele->superCluster().isNull()))
     { electrons->push_back(*ele) ; }
    else
     { LogDebug("GsfElectronCoreProducer")<<"GsfTrack with no associated CaloCluster." ; }
    delete ele ;
   }

  // clone and complete ecal driven electrons
  const GsfElectronCoreCollection * edCoresCollection = edCoresH.product() ;
  GsfElectronCoreCollection::const_iterator edCoreIter ;
  for
   ( edCoreIter = edCoresCollection->begin() ;
     edCoreIter != edCoresCollection->end() ;
     edCoreIter++ )
   {
    GsfElectronCore * eleCore = edCoreIter->clone() ;
    const GsfTrackRef & gsfTrackRef = eleCore->gsfTrack() ;
    eleCore->setPflowSuperCluster((*pfClusterTracksH)[gsfTrackRef]) ;
    electrons->push_back(*eleCore) ;
    delete eleCore ;
   }

  event.put(electrons) ;
 }

GsfElectronCoreProducer::~GsfElectronCoreProducer()
 {}

