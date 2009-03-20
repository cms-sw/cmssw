
#include "GsfElectronCoreProducer.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCoreFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"

using namespace reco ;

GsfElectronCoreProducer::GsfElectronCoreProducer( const edm::ParameterSet & config )
 {
  produces<GsfElectronCoreCollection>() ;
  gsfTracksTag_ = config.getParameter<edm::InputTag>("gsfTracks") ;
  pfSuperClustersTag_ = config.getParameter<edm::InputTag>("pfSuperClusters") ;
 }

void GsfElectronCoreProducer::produce( edm::Event & event, const edm::EventSetup & setup )
 {
  // output
  std::auto_ptr<GsfElectronCoreCollection> electrons(new GsfElectronCoreCollection) ;

  // input
  edm::Handle<GsfTrackCollection> gsfTracksH ;
  event.getByLabel(gsfTracksTag_,gsfTracksH) ;
//  edm::Handle<SuperClusterCollection> pfSuperClustersH ;
//  event.getByLabel(pfSuperClustersTag_,pfSuperClustersH) ;
//  edm::Handle<???> pfSuperClustersTracksAssocH ;
//  event.getByLabel(pfSuperClustersTag_,pfSuperClustersTracksAssocH) ;

  // loop
  const GsfTrackCollection * gsfTrackCollection = gsfTracksH.product() ;
  for ( unsigned int i=0 ; i<gsfTrackCollection->size() ; ++i )
   {
    //const GsfTrack & t=(*gsfTrackCollection)[i] ;
    const GsfTrackRef gsfTrackRef = edm::Ref<GsfTrackCollection>(gsfTracksH,i) ;
    GsfElectronCore * ele = new GsfElectronCore(gsfTrackRef) ;
    if (ele->isEcalDriven())
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
    if (ele->isTrackerDriven())
     {
      // eventual pflow super cluster.
      //...
      //ele->setPflowSuperCluster(??) ;
     }
    electrons->push_back(*ele) ;
   }
  event.put(electrons) ;
 }

GsfElectronCoreProducer::~GsfElectronCoreProducer()
 {}



