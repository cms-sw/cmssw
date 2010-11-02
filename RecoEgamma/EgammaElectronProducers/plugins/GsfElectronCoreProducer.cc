
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
  ctfTracksTag_ = config.getParameter<edm::InputTag>("ctfTracks");
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
  edm::Handle<TrackCollection> ctfTracksH ;
  event.getByLabel(ctfTracksTag_,ctfTracksH) ;
  edm::Handle<SuperClusterCollection> pfClustersH;
  event.getByLabel(pfSuperClustersTag_,pfClustersH);
  edm::Handle<edm::ValueMap<reco::SuperClusterRef> > pfClusterTracksH;
  event.getByLabel(pfSuperClusterTrackMapTag_,pfClusterTracksH);

  // loop
  const GsfTrackCollection * gsfTrackCollection = gsfTracksH.product() ;
  for ( unsigned int i=0 ; i<gsfTrackCollection->size() ; ++i )
   {
    const GsfTrackRef gsfTrackRef = edm::Ref<GsfTrackCollection>(gsfTracksH,i) ;
    GsfElectronCore * ele = new GsfElectronCore(gsfTrackRef) ;

    std::pair<TrackRef,float> ctfpair = getCtfTrackRef(gsfTrackRef,ctfTracksH) ;
    ele->setCtfTrack(ctfpair.first,ctfpair.second) ;

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


//=======================================================================================
// Code from Puneeth Kalavase
//=======================================================================================

std::pair<TrackRef,float> GsfElectronCoreProducer::getCtfTrackRef
 ( const GsfTrackRef& gsfTrackRef, edm::Handle<reco::TrackCollection> ctfTracksH )
 {
  float maxFracShared = 0;
  TrackRef ctfTrackRef = TrackRef() ;
  const TrackCollection * ctfTrackCollection = ctfTracksH.product() ;

  // get the Hit Pattern for the gsfTrack
  const HitPattern& gsfHitPattern = gsfTrackRef->hitPattern();

  unsigned int counter ;
  TrackCollection::const_iterator ctfTkIter ;
  for ( ctfTkIter = ctfTrackCollection->begin() , counter = 0 ;
        ctfTkIter != ctfTrackCollection->end() ; ctfTkIter++, counter++ )
   {

    double dEta = gsfTrackRef->eta() - ctfTkIter->eta();
    double dPhi = gsfTrackRef->phi() - ctfTkIter->phi();
    double pi = acos(-1.);
    if(fabs(dPhi) > pi) dPhi = 2*pi - fabs(dPhi);

    // dont want to look at every single track in the event!
    if(sqrt(dEta*dEta + dPhi*dPhi) > 0.3) continue;

    unsigned int shared = 0 ;
    int gsfHitCounter = 0 ;
    int numGsfInnerHits = 0 ;
    int numCtfInnerHits = 0 ;
    // get the CTF Track Hit Pattern
    const HitPattern& ctfHitPattern = ctfTkIter->hitPattern() ;

    trackingRecHit_iterator elHitsIt ;
    for ( elHitsIt = gsfTrackRef->recHitsBegin() ;
          elHitsIt != gsfTrackRef->recHitsEnd() ;
          elHitsIt++, gsfHitCounter++ )
     {
      if(!((**elHitsIt).isValid()))  //count only valid Hits
       { continue ; }

      // look only in the pixels/TIB/TID
      uint32_t gsfHit = gsfHitPattern.getHitPattern(gsfHitCounter) ;
      if (!(gsfHitPattern.pixelHitFilter(gsfHit) ||
          gsfHitPattern.stripTIBHitFilter(gsfHit) ||
          gsfHitPattern.stripTIDHitFilter(gsfHit) ) )
       { continue ; }

      numGsfInnerHits++ ;

      int ctfHitsCounter = 0 ;
      numCtfInnerHits = 0 ;
      trackingRecHit_iterator ctfHitsIt ;
      for ( ctfHitsIt = ctfTkIter->recHitsBegin() ;
            ctfHitsIt != ctfTkIter->recHitsEnd() ;
            ctfHitsIt++, ctfHitsCounter++ )
       {
        if(!((**ctfHitsIt).isValid())) //count only valid Hits!
         { continue ; }

      uint32_t ctfHit = ctfHitPattern.getHitPattern(ctfHitsCounter);
      if( !(ctfHitPattern.pixelHitFilter(ctfHit) ||
            ctfHitPattern.stripTIBHitFilter(ctfHit) ||
            ctfHitPattern.stripTIDHitFilter(ctfHit) ) )
       { continue ; }

      numCtfInnerHits++ ;

        if( (**elHitsIt).sharesInput(&(**ctfHitsIt),TrackingRecHit::all) )
         {
          shared++ ;
          break ;
         }

       } //ctfHits iterator

     } //gsfHits iterator

    if ((numGsfInnerHits==0)||(numCtfInnerHits==0))
     { continue ; }

    if ( static_cast<float>(shared)/std::min(numGsfInnerHits,numCtfInnerHits) > maxFracShared )
     {
      maxFracShared = static_cast<float>(shared)/std::min(numGsfInnerHits, numCtfInnerHits);
      ctfTrackRef = TrackRef(ctfTracksH,counter);
     }

   } //ctfTrack iterator

  return make_pair(ctfTrackRef,maxFracShared) ;
 }



