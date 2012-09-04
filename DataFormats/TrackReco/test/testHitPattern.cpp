#include "DataFormats/TrackReco/interface/HitPattern.h"



int main() {

  unsigned char hit[reco::HitPattern::MaxHits];

  int nhit;
  auto unpack =[&hit,&nhit](uint32_t pattern) {
    hit[nhit++]= 255&&(pattern>>3);
    // buuble sort
    if (nhit>1)
    for (auto h=hit+nhit-1; h!=hit+1; --h) {
      if ( (*(h-1)) <= (*h)) break;
      std::swap(*(h-1),(*h));
    }
  }

  reco::HitPattern hp;

  hp.call(reco::HitPattern::validHitFilter,unpack);

  return nhit=0;
}
