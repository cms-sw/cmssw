
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
//   desc.add<edm::InputTag>("pflowDrivenGsfElectronCoresTag",edm::InputTag("pflowGsfElectronCores")) ;
//   desc.add<edm::InputTag>("pfSuperClusters",edm::InputTag("pfElectronTranslator:pf")) ;
//   desc.add<edm::InputTag>("pfSuperClusterTrackMap",edm::InputTag("pfElectronTranslator:pf")) ;
//   descriptions.add("produceGsfElectronCores",desc) ;
//  }

GsfElectronCoreProducer::GsfElectronCoreProducer( const edm::ParameterSet & config )
 : GsfElectronCoreBaseProducer(config)
 {
   edCoresTag_ = consumes<reco::GsfElectronCoreCollection>(config.getParameter<edm::InputTag>("ecalDrivenGsfElectronCoresTag"));
   pfCoresTag_ = consumes<reco::GsfElectronCoreCollection>(config.getParameter<edm::InputTag>("pflowGsfElectronCoresTag"));
//  pfSuperClustersTag_ = config.getParameter<edm::InputTag>("pfSuperClusters") ;
//  pfSuperClusterTrackMapTag_ = config.getParameter<edm::InputTag>("pfSuperClusterTrackMap") ;
 }

void GsfElectronCoreProducer::produce( edm::Event & event, const edm::EventSetup & setup )
 {
  // base input
  GsfElectronCoreBaseProducer::initEvent(event,setup) ;

  // transient output
  std::list<GsfElectronCore *> electrons ;

  // event input
  event.getByToken(edCoresTag_,edCoresH_) ;
  event.getByToken(pfCoresTag_,pfCoresH_) ;
//  event.getByToken(pfSuperClustersTag_,pfClustersH_) ;
//  event.getByToken(pfSuperClusterTrackMapTag_,pfClusterTracksH_) ;

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
      produceTrackerDrivenCore(gsfTrackRef,electrons) ;
     }
   }
  else
   {
    const GsfTrackCollection * gsfTrackCollection = gsfTracksH_.product() ;
    for ( unsigned int i=0 ; i<gsfTrackCollection->size() ; ++i )
     {
      const GsfTrackRef gsfTrackRef = edm::Ref<GsfTrackCollection>(gsfTracksH_,i) ;
      produceTrackerDrivenCore(gsfTrackRef,electrons) ;
     }
   }

  // clone ecal driven electrons
  const GsfElectronCoreCollection * edCoresCollection = edCoresH_.product() ;
  GsfElectronCoreCollection::const_iterator edCoreIter ;
  for
   ( edCoreIter = edCoresCollection->begin() ;
     edCoreIter != edCoresCollection->end() ;
     edCoreIter++ )
   { electrons.push_back(edCoreIter->clone()) ; }

  // add pflow info
  const GsfElectronCoreCollection * pfCoresCollection = pfCoresH_.product() ;
  GsfElectronCoreCollection::const_iterator pfCoreIter ;
  std::list<GsfElectronCore *>::iterator eleCore ;
  bool found ;
  for ( eleCore = electrons.begin() ; eleCore != electrons.end() ; eleCore++ )
   {
//    (*eleCore)->setParentSuperCluster((*pfClusterTracksH_)[(*eleCore)->gsfTrack()]) ;
    found = false ;
    for
     ( pfCoreIter = pfCoresCollection->begin() ;
       pfCoreIter != pfCoresCollection->end() ;
       pfCoreIter++ )
     {
      if (pfCoreIter->gsfTrack()==(*eleCore)->gsfTrack())
       {
        if (found)
         {
          edm::LogWarning("GsfElectronCoreProducer")<<"associated pfGsfElectronCore already found" ;
         }
        else
         {
          found = true ;
          (*eleCore)->setParentSuperCluster(pfCoreIter->parentSuperCluster()) ;
         }
       }
     }
   }

  // store
  auto collection = std::make_unique<GsfElectronCoreCollection>();
  for ( eleCore = electrons.begin() ; eleCore != electrons.end() ; eleCore++ )
   {
    if ((*eleCore)->superCluster().isNull())
     { LogDebug("GsfElectronCoreProducer")<<"GsfTrack with no associated SuperCluster at all." ; }
    else
     { collection->push_back(**eleCore) ; }
    delete (*eleCore) ;
   }
  event.put(std::move(collection));
 }

void GsfElectronCoreProducer::produceTrackerDrivenCore( const GsfTrackRef & gsfTrackRef, std::list<GsfElectronCore *> & electrons )
 {
  GsfElectronCore * eleCore = new GsfElectronCore(gsfTrackRef) ;
  if (eleCore->ecalDrivenSeed())
   { delete eleCore ; return ; }
  GsfElectronCoreBaseProducer::fillElectronCore(eleCore) ;
  electrons.push_back(eleCore) ;
 }

GsfElectronCoreProducer::~GsfElectronCoreProducer()
 {}

