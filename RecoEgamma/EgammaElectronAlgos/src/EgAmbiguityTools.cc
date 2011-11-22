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
 { return (std::abs(e1->eSuperClusterOverP()-1)<std::abs(e2->eSuperClusterOverP()-1)) ; }

bool isInnerMost::operator()
 ( const reco::GsfElectron * e1, const reco::GsfElectron * e2 )
 {
  reco::HitPattern gsfHitPattern1 = e1->gsfTrack()->hitPattern() ;
  reco::HitPattern gsfHitPattern2 = e2->gsfTrack()->hitPattern() ;

  // retreive first valid hit
  int gsfHitCounter1 = 0 ;
  trackingRecHit_iterator elHitsIt1 ;
  for
   ( elHitsIt1 = e1->gsfTrack()->recHitsBegin() ;
     elHitsIt1 != e1->gsfTrack()->recHitsEnd() ;
     elHitsIt1++, gsfHitCounter1++ )
   { if (((**elHitsIt1).isValid())) break ; }

  int gsfHitCounter2 = 0 ;
  trackingRecHit_iterator elHitsIt2 ;
  for
   ( elHitsIt2 = e2->gsfTrack()->recHitsBegin() ;
     elHitsIt2 != e2->gsfTrack()->recHitsEnd() ;
     elHitsIt2++, gsfHitCounter2++ )
   { if (((**elHitsIt2).isValid())) break ; }

  uint32_t gsfHit1 = gsfHitPattern1.getHitPattern(gsfHitCounter1) ;
  uint32_t gsfHit2 = gsfHitPattern2.getHitPattern(gsfHitCounter2) ;
  if (gsfHitPattern1.getSubStructure(gsfHit1)!=gsfHitPattern2.getSubStructure(gsfHit2))
   { return (gsfHitPattern1.getSubStructure(gsfHit1)<gsfHitPattern2.getSubStructure(gsfHit2)) ; }
  else if (gsfHitPattern1.getLayer(gsfHit1)!=gsfHitPattern2.getLayer(gsfHit2))
   { return (gsfHitPattern1.getLayer(gsfHit1)<gsfHitPattern2.getLayer(gsfHit2)) ; }
  else
   { return isBetter(e1,e2) ; }
 }

int sharedHits( const GsfTrackRef & gsfTrackRef1, const GsfTrackRef & gsfTrackRef2 )
 {
  //get the Hit Pattern for the gsfTracks
  const HitPattern & gsfHitPattern1 = gsfTrackRef1->hitPattern() ;
  const HitPattern & gsfHitPattern2 = gsfTrackRef2->hitPattern() ;

  unsigned int shared = 0;

  int gsfHitCounter1 = 0;
  for(trackingRecHit_iterator elHitsIt1 = gsfTrackRef1->recHitsBegin();
      elHitsIt1 != gsfTrackRef1->recHitsEnd(); elHitsIt1++, gsfHitCounter1++) {
    if(!((**elHitsIt1).isValid()))  //count only valid Hits
      continue;
    //if (gsfHitCounter1>1) continue; // test only the first hit of the track 1
    uint32_t gsfHit = gsfHitPattern1.getHitPattern(gsfHitCounter1);
    if(!(gsfHitPattern1.pixelHitFilter(gsfHit)
	 || gsfHitPattern1.stripTIBHitFilter(gsfHit)
	 || gsfHitPattern1.stripTOBHitFilter(gsfHit)
	 || gsfHitPattern1.stripTECHitFilter(gsfHit)
	 || gsfHitPattern1.stripTIDHitFilter(gsfHit) )
    ) continue;
    int gsfHitsCounter2 = 0;
    for(trackingRecHit_iterator gsfHitsIt2 = gsfTrackRef2->recHitsBegin();
        gsfHitsIt2 != gsfTrackRef2->recHitsEnd(); gsfHitsIt2++, gsfHitsCounter2++) {
      if(!((**gsfHitsIt2).isValid())) //count only valid Hits!
	continue;

      uint32_t gsfHit2 = gsfHitPattern2.getHitPattern(gsfHitsCounter2);
      if(!(gsfHitPattern2.pixelHitFilter(gsfHit2)
	 || gsfHitPattern2.stripTIBHitFilter(gsfHit2)
	 || gsfHitPattern2.stripTOBHitFilter(gsfHit2)
	 || gsfHitPattern2.stripTECHitFilter(gsfHit2)
	 || gsfHitPattern2.stripTIDHitFilter(gsfHit2) )
      ) continue;
      if( (**elHitsIt1).sharesInput(&(**gsfHitsIt2), TrackingRecHit::some) ) {
//        if (comp.equals(&(**elHitsIt1),&(**gsfHitsIt2))) {
        //std::cout << "found shared hit " << gsfHit2 << std::endl;
  	shared++;
      }
    }//gsfHits2 iterator
  }//gsfHits1 iterator

  //std::cout << "[sharedHits] number of shared hits " << shared << std::endl;
  return shared;

}

int sharedDets(const GsfTrackRef& gsfTrackRef1, const
 GsfTrackRef& gsfTrackRef2 ) {

  //get the Hit Pattern for the gsfTracks
  const HitPattern& gsfHitPattern1 = gsfTrackRef1->hitPattern();
  const HitPattern& gsfHitPattern2 = gsfTrackRef2->hitPattern();

  unsigned int shared = 0;

  int gsfHitCounter1 = 0;
  for(trackingRecHit_iterator elHitsIt1 = gsfTrackRef1->recHitsBegin();
      elHitsIt1 != gsfTrackRef1->recHitsEnd(); elHitsIt1++, gsfHitCounter1++) {
    if(!((**elHitsIt1).isValid()))  //count only valid Hits
      continue;
    //if (gsfHitCounter1>1) continue; // to test only the first hit of the track 1
    uint32_t gsfHit = gsfHitPattern1.getHitPattern(gsfHitCounter1);
    if(!(gsfHitPattern1.pixelHitFilter(gsfHit)
	 || gsfHitPattern1.stripTIBHitFilter(gsfHit)
	 || gsfHitPattern1.stripTOBHitFilter(gsfHit)
	 || gsfHitPattern1.stripTECHitFilter(gsfHit)
	 || gsfHitPattern1.stripTIDHitFilter(gsfHit) )
    ) continue;
    int gsfHitsCounter2 = 0;
    for(trackingRecHit_iterator gsfHitsIt2 = gsfTrackRef2->recHitsBegin();
        gsfHitsIt2 != gsfTrackRef2->recHitsEnd(); gsfHitsIt2++, gsfHitsCounter2++) {
      if(!((**gsfHitsIt2).isValid())) //count only valid Hits!
	continue;

      uint32_t gsfHit2 = gsfHitPattern2.getHitPattern(gsfHitsCounter2);
      if(!(gsfHitPattern2.pixelHitFilter(gsfHit2)
	   || gsfHitPattern2.stripTIBHitFilter(gsfHit2)
	   || gsfHitPattern1.stripTOBHitFilter(gsfHit2)
	   || gsfHitPattern2.stripTECHitFilter(gsfHit2)
	   || gsfHitPattern2.stripTIDHitFilter(gsfHit2) )
      ) continue;
      if ((**elHitsIt1).geographicalId() == (**gsfHitsIt2).geographicalId()) shared++;
    }//gsfHits2 iterator
  }//gsfHits1 iterator

  //std::cout << "[sharedHits] number of shared dets " << shared << std::endl;
  //return shared/min(gsfTrackRef1->numberOfValidHits(),gsfTrackRef2->numberOfValidHits());
  return shared;

}

float sharedEnergy(const CaloCluster *clu1, const CaloCluster *clu2,
       edm::Handle<EcalRecHitCollection> & reducedEBRecHits,
       edm::Handle<EcalRecHitCollection> & reducedEERecHits ) {

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
	if ((itt=reducedEBRecHits->find((*ih1).first))!=reducedEBRecHits->end())
	fractionShared += itt->energy();
      } else if ((*ih1).first.subdetId() == EcalEndcap) {
	if ((itt=reducedEERecHits->find((*ih1).first))!=reducedEERecHits->end())
	fractionShared += itt->energy();
      }

    }
  }

  //std::cout << "[sharedEnergy] shared energy /min(energy1,energy2) " << fractionShared << std::endl;
  return fractionShared;

}

float sharedEnergy(const SuperClusterRef& sc1, const SuperClusterRef& sc2,
       edm::Handle<EcalRecHitCollection> & reducedEBRecHits,
       edm::Handle<EcalRecHitCollection> & reducedEERecHits ) {

  double energyShared = 0;
  for(CaloCluster_iterator icl1=sc1->clustersBegin();icl1!=sc1->clustersEnd(); icl1++) {
    for(CaloCluster_iterator icl2=sc2->clustersBegin();icl2!=sc2->clustersEnd(); icl2++) {
      energyShared += sharedEnergy(&(**icl1),&(**icl2),reducedEBRecHits,reducedEERecHits );
    }
  }
  return energyShared;

}

}
