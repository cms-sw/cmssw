#define private public
#include "DataFormats/TrackReco/interface/HitPattern.h"
#undef private

#include <random>
#include <algorithm>


#include<cstdio>
#include<iostream>
int main() {


  reco::HitPattern hp1;
  reco::HitPattern hp2;
  std::mt19937 eng;
  std::uniform_int_distribution<int> ugen(1,255);
 
  hp1.setHitPattern(0, (121<<3));
  hp2.setHitPattern(0, (121<<3));
  hp1.setHitPattern(1, (121<<3));
  hp2.setHitPattern(1, (125<<3));
  hp1.setHitPattern(2, (121<<3));
  hp2.setHitPattern(2, (121<<3));

  for (int i=3; i!=20;++i) {
    if (i%7==1) { 
      hp1.setHitPattern(i, (123<<3)+1); // invalid
    }
    if (i%3==1) { 
      int p = ugen(eng);
      hp1.setHitPattern(i,p <<3);
      hp2.setHitPattern(i,p <<3);
    } else{
      hp1.setHitPattern(i,ugen(eng) <<3);
      hp2.setHitPattern(i,ugen(eng) <<3);
    }
  }

  for (int i=0; i!=15;++i) {
    printf("%d,%d ",hp1.getHitPattern(i)>>3,
	   hp2.getHitPattern(i)>>3
	   );
  }
  printf("\n");


  reco::PatternSet<15> p1(hp1), p2(hp2);

  reco::PatternSet<15> comm = reco::commonHits(p1,p2);
  std::cout << "common " << comm.size() << std::endl;
  for (auto p:comm) printf("%d ",int(p));
  printf("\n");

  assert(p1.size()==15);
  assert(p2.size()==15);
  for (int i=0; i!=14;++i) {
    printf("%d,%d ",int(p1[i]),int(p2[i]));
    assert(p1[i]!=0);
    assert(p2[i]!=0);
    assert(p1[i]<=p1[i+1]);
    assert(p2[i]<=p2[i+1]);
  }
  printf("\n");
  return 0;
}
