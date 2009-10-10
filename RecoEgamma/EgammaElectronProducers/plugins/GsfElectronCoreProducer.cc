
#include "GsfElectronCoreProducer.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCoreFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include <map>

using namespace reco ;

GsfElectronCoreProducer::GsfElectronCoreProducer( const edm::ParameterSet & config )
 {
  produces<GsfElectronCoreCollection>() ;
  gsfTracksTag_ = config.getParameter<edm::InputTag>("gsfTracks") ;
  pfSuperClustersTag_ = config.getParameter<edm::InputTag>("pfSuperClusters") ;
  pfSuperClusterTrackMapTag_ = config.getParameter<edm::InputTag>("pfSuperClusterTrackMap") ;
 }

void GsfElectronCoreProducer::produce( edm::Event & event, const edm::EventSetup & setup )
 {
  // output
  std::auto_ptr<GsfElectronCoreCollection> electrons(new GsfElectronCoreCollection) ;

  // input
  edm::Handle<GsfTrackCollection> gsfTracksH ;
  event.getByLabel(gsfTracksTag_,gsfTracksH) ;
  edm::Handle<SuperClusterCollection> pfClustersH;
  event.getByLabel(pfSuperClustersTag_,pfClustersH);
  edm::Handle<edm::ValueMap<reco::SuperClusterRef> > pfClusterTracksH;
  event.getByLabel(pfSuperClusterTrackMapTag_,pfClusterTracksH);

  // loop
  const GsfTrackCollection * gsfTrackCollection = gsfTracksH.product() ;
  for ( unsigned int i=0 ; i<gsfTrackCollection->size() ; ++i )
   {
    //const GsfTrack & t=(*gsfTrackCollection)[i] ;
    const GsfTrackRef gsfTrackRef = edm::Ref<GsfTrackCollection>(gsfTracksH,i) ;
    GsfElectronCore * ele = new GsfElectronCore(gsfTrackRef) ;
    if (ele->ecalDrivenSeed())
     {
      edm::RefToBase<TrajectorySeed> seed = gsfTrackRef->extra()->seedRef() ;
      ElectronSeedRef elseed = seed.castTo<ElectronSeedRef>() ;
      edm::RefToBase<CaloCluster> caloCluster = elseed->caloCluster() ;
      SuperClusterRef scRef = caloCluster.castTo<SuperClusterRef>() ;
      if (!scRef.isNull())
       { ele->setSuperCluster(scRef) ; }
      else
       { edm::LogWarning("GsfElectronCoreProducer")<<"Seed CaloCluster is not a SuperCluster, unexpected..." ; }
     }
    //if (ele->trackerDrivenSeed())
    // {
    //  // eventual pflow super cluster.
    //  SuperClusterRef pfscRef = (*pfClusterTracksH)[gsfTrackRef];
    //  ele->setPflowSuperCluster(pfscRef) ;
     //}
    ele->setPflowSuperCluster((*pfClusterTracksH)[gsfTrackRef]) ;
    if (!(ele->superCluster().isNull()))
     { electrons->push_back(*ele) ; }
    else
     { LogDebug("GsfElectronCoreProducer")<<"GsfTrack with no associated CaloCluster." ; }
    delete ele ;
   }
  event.put(electrons) ;
 }

GsfElectronCoreProducer::~GsfElectronCoreProducer()
 {}



