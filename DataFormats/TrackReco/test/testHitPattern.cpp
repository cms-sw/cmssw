#include "DataFormats/TrackReco/interface/HitPattern.h"



int main() {

  unsigned char hit[reco::HitPattern::MaxHits];

  int nhit;
  auto unpack =[&hit,&nhit](uint32_t pattern) {
    hit[nhit++]= 255&&(pattern>>3);
  };

  reco::HitPattern hp;

  hp.call(reco::HitPattern::validHitFilter,unpack);

  return nhit=0;
}
