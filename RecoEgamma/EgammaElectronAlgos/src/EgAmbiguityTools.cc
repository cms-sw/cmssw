#include "RecoEgamma/EgammaElectronAlgos/interface/EgAmbiguityTools.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/EgAmbiguityTools.h"
//#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronHcalHelper.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateTransform.h"

#include <sstream>
#include <algorithm>

using namespace edm ;
using namespace std ;
using namespace reco ;

namespace EgAmbiguityTools
{

bool isBetter( const reco::GsfElectron * e1, const reco::GsfElectron * e2 )
 { return (std::abs(e1->eSuperClusterOverP()-1) < std::abs(e2->eSuperClusterOverP() - 1)) ; }

bool isInnerMost::operator()(const reco::GsfElectron *e1, const reco::GsfElectron *e2)
{
    // retreive first valid hit
    int gsfHitCounter1 = 0 ;
    trackingRecHit_iterator elHitsIt1 ;
    for(elHitsIt1 = e1->gsfTrack()->recHitsBegin();
            elHitsIt1 != e1->gsfTrack()->recHitsEnd();
            elHitsIt1++, gsfHitCounter1++)
    { 
        if ((**elHitsIt1).isValid()){
            break;
        }
    }

    int gsfHitCounter2 = 0 ;
    trackingRecHit_iterator elHitsIt2 ;
    for(elHitsIt2 = e2->gsfTrack()->recHitsBegin();
            elHitsIt2 != e2->gsfTrack()->recHitsEnd();
            elHitsIt2++, gsfHitCounter2++ )
    {
        if ((**elHitsIt2).isValid()){
            break;
        }
    }

    uint32_t gsfHit1 = e1->gsfTrack()->hitPattern().getHitPattern(HitPattern::TRACK_HITS, gsfHitCounter1);
    uint32_t gsfHit2 = e2->gsfTrack()->hitPattern().getHitPattern(HitPattern::TRACK_HITS, gsfHitCounter2);

    if (HitPattern::getSubStructure(gsfHit1) != HitPattern::getSubStructure(gsfHit2)){
        return (HitPattern::getSubStructure(gsfHit1) < HitPattern::getSubStructure(gsfHit2));
    }else if (HitPattern::getLayer(gsfHit1) != HitPattern::getLayer(gsfHit2)){
        return (HitPattern::getLayer(gsfHit1) < HitPattern::getLayer(gsfHit2));
    }else{
        return isBetter(e1, e2);
    }
}

int sharedHits(const GsfTrackRef & gsfTrackRef1, const GsfTrackRef & gsfTrackRef2 )
{
  //get the Hit Pattern for the gsfTracks
  const HitPattern &gsfHitPattern1 = gsfTrackRef1->hitPattern();
  const HitPattern &gsfHitPattern2 = gsfTrackRef2->hitPattern();

  unsigned int shared = 0;

  int gsfHitCounter1 = 0;
  for(trackingRecHit_iterator elHitsIt1 = gsfTrackRef1->recHitsBegin();
          elHitsIt1 != gsfTrackRef1->recHitsEnd(); elHitsIt1++, gsfHitCounter1++) {
      if(!(**elHitsIt1).isValid()){
          //count only valid Hits
          continue;
      }
      //if (gsfHitCounter1>1) continue; // test only the first hit of the track 1
      uint32_t gsfHit = gsfHitPattern1.getHitPattern(HitPattern::TRACK_HITS, gsfHitCounter1);
      if(!(HitPattern::pixelHitFilter(gsfHit)
                  || HitPattern::stripTIBHitFilter(gsfHit)
                  || HitPattern::stripTOBHitFilter(gsfHit)
                  || HitPattern::stripTECHitFilter(gsfHit)
                  || HitPattern::stripTIDHitFilter(gsfHit))){
          continue;
      }

      int gsfHitsCounter2 = 0;
      for(trackingRecHit_iterator gsfHitsIt2 = gsfTrackRef2->recHitsBegin();
              gsfHitsIt2 != gsfTrackRef2->recHitsEnd(); gsfHitsIt2++, gsfHitsCounter2++) {
          if(!(**gsfHitsIt2).isValid()){
              //count only valid Hits
              continue;
          }
          uint32_t gsfHit2 = gsfHitPattern2.getHitPattern(HitPattern::TRACK_HITS, gsfHitsCounter2);
          if(!(HitPattern::pixelHitFilter(gsfHit2)
                      || HitPattern::stripTIBHitFilter(gsfHit2)
                      || HitPattern::stripTOBHitFilter(gsfHit2)
                      || HitPattern::stripTECHitFilter(gsfHit2)
                      || HitPattern::stripTIDHitFilter(gsfHit2))){
              continue;
          }
          if((**elHitsIt1).sharesInput(&(**gsfHitsIt2), TrackingRecHit::some)) {
              //if (comp.equals(&(**elHitsIt1),&(**gsfHitsIt2))) {
              ////std::cout << "found shared hit " << gsfHit2 << std::endl;
              shared++;
          }
      }//gsfHits2 iterator
  }//gsfHits1 iterator

  //std::cout << "[sharedHits] number of shared hits " << shared << std::endl;
  return shared;
}

int sharedDets(const GsfTrackRef& gsfTrackRef1, const GsfTrackRef& gsfTrackRef2 ) 
{
    //get the Hit Pattern for the gsfTracks
    const HitPattern &gsfHitPattern1 = gsfTrackRef1->hitPattern();
    const HitPattern &gsfHitPattern2 = gsfTrackRef2->hitPattern();

    unsigned int shared = 0;

    int gsfHitCounter1 = 0;
    for(trackingRecHit_iterator elHitsIt1 = gsfTrackRef1->recHitsBegin();
            elHitsIt1 != gsfTrackRef1->recHitsEnd(); elHitsIt1++, gsfHitCounter1++) {
        if(!((**elHitsIt1).isValid())){
            //count only valid Hits
            continue;
        }
        //if (gsfHitCounter1>1) continue; // to test only the first hit of the track 1
        uint32_t gsfHit = gsfHitPattern1.getHitPattern(HitPattern::TRACK_HITS, gsfHitCounter1);
        if(!(HitPattern::pixelHitFilter(gsfHit)
                    || HitPattern::stripTIBHitFilter(gsfHit)
                    || HitPattern::stripTOBHitFilter(gsfHit)
                    || HitPattern::stripTECHitFilter(gsfHit)
                    || HitPattern::stripTIDHitFilter(gsfHit)))
        {
            continue;
        }
        
        int gsfHitsCounter2 = 0;
        for(trackingRecHit_iterator gsfHitsIt2 = gsfTrackRef2->recHitsBegin();
                gsfHitsIt2 != gsfTrackRef2->recHitsEnd(); gsfHitsIt2++, gsfHitsCounter2++) {
            if(!((**gsfHitsIt2).isValid())){
                //count only valid Hits!
                continue;
            }

            uint32_t gsfHit2 = gsfHitPattern2.getHitPattern(HitPattern::TRACK_HITS, gsfHitsCounter2);
            if(!(HitPattern::pixelHitFilter(gsfHit2)
                        || HitPattern::stripTIBHitFilter(gsfHit2)
                        || HitPattern::stripTOBHitFilter(gsfHit2)
                        || HitPattern::stripTECHitFilter(gsfHit2)
                        || HitPattern::stripTIDHitFilter(gsfHit2) )
              ) continue;
            if ((**elHitsIt1).geographicalId() == (**gsfHitsIt2).geographicalId()) shared++;
        }//gsfHits2 iterator
    }//gsfHits1 iterator

    //std::cout << "[sharedHits] number of shared dets " << shared << std::endl;
    //return shared/min(gsfTrackRef1->numberOfValidHits(),gsfTrackRef2->numberOfValidHits());
    return shared;

}

float sharedEnergy(const CaloCluster *clu1, const CaloCluster *clu2,
       edm::Handle<EcalRecHitCollection> & barrelRecHits,
       edm::Handle<EcalRecHitCollection> & endcapRecHits ) {

  double fractionShared = 0;

  std::vector< std::pair<DetId, float> > v_id1 = clu1->hitsAndFractions();
  std::vector< std::pair<DetId, float> > v_id2 = clu2->hitsAndFractions();
  std::vector< std::pair<DetId, float> >::iterator ih1;
  std::vector< std::pair<DetId, float> >::iterator ih2;

  for(ih1 = v_id1.begin();ih1 != v_id1.end(); ih1++) {

    for(ih2 = v_id2.begin();ih2 != v_id2.end(); ih2++) {

      if ( (*ih1).first != (*ih2).first ) continue;

      // here we have common Xtal id
      EcalRecHitCollection::const_iterator itt;
      if ((*ih1).first.subdetId() == EcalBarrel) {
	if ((itt=barrelRecHits->find((*ih1).first))!=barrelRecHits->end())
	fractionShared += itt->energy();
      } else if ((*ih1).first.subdetId() == EcalEndcap) {
	if ((itt=endcapRecHits->find((*ih1).first))!=endcapRecHits->end())
	fractionShared += itt->energy();
      }

    }
  }

  //std::cout << "[sharedEnergy] shared energy /min(energy1,energy2) " << fractionShared << std::endl;
  return fractionShared;

}

float sharedEnergy(const SuperClusterRef& sc1, const SuperClusterRef& sc2,
       edm::Handle<EcalRecHitCollection> & barrelRecHits,
       edm::Handle<EcalRecHitCollection> & endcapRecHits ) {

  double energyShared = 0;
  for(CaloCluster_iterator icl1=sc1->clustersBegin();icl1!=sc1->clustersEnd(); icl1++) {
    for(CaloCluster_iterator icl2=sc2->clustersBegin();icl2!=sc2->clustersEnd(); icl2++) {
      energyShared += sharedEnergy(&(**icl1),&(**icl2),barrelRecHits,endcapRecHits );
    }
  }
  return energyShared;

}

}
