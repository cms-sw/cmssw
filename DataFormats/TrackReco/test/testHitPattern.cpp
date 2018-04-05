#include "DataFormats/TrackReco/interface/HitPattern.h"

#include <random>
#include <algorithm>
#include <cstdio>
#include <iostream>

namespace test {
  namespace TestHitPattern {
    using namespace reco;

    int test()
    {

    {
      const uint32_t radial_detids[] = { 402666125,//TID r1
    				     402668833,//TID r2
    				     402673476,//TID r3
    				     470066725,//TEC r1
    				     470390853,//TEC r2
    				     470114664,//TEC r3
    				     470131344,//TEC r4
    				     470079661,//TEC r5
    				     470049476,//TEC r6
    				     470045428}; //TEC r7
    
       HitPattern hp;
       auto i=0;
       for (auto id : radial_detids) { hp.appendHit(id,(i++ == 1) ? TrackingRecHit::missing : TrackingRecHit::valid);}
       hp.appendHit(radial_detids[2],TrackingRecHit::missing);
       hp.appendHit(radial_detids[8],TrackingRecHit::missing);
    
    
       std::cout << hp.numberOfValidTrackerHits() << ' ' << hp.numberOfValidPixelHits() << ' ' <<	hp.numberOfValidStripHits() << std::endl;
       std::cout << hp.pixelLayersWithMeasurement() << ' ' << hp.stripLayersWithMeasurement() << std::endl;
       std::cout << hp.numberOfValidStripLayersWithMonoAndStereo() << std::endl;
       std::cout <<	hp.pixelLayersWithoutMeasurement(HitPattern::TRACK_HITS) << ' ' << hp.stripLayersWithoutMeasurement(HitPattern::TRACK_HITS) << std::endl;
    
    }
    
    
    
        HitPattern hp1;
        HitPattern hp2;
        std::mt19937 eng;
        std::uniform_int_distribution<int> ugen(1, 255);
    
        hp1.insertTrackHit(121 << 3);
        hp2.insertTrackHit(121 << 3);
        hp1.insertTrackHit(121 << 3);
        hp2.insertTrackHit(125 << 3);
        hp1.insertTrackHit(121 << 3);
        hp2.insertTrackHit(121 << 3);
    
        for (int i = 3; i != 20; ++i) {
            if (i % 7 == 1) {
                hp1.insertTrackHit((123 << 3) + 1); // invalid
            }
            if (i % 3 == 1) {
                int p = ugen(eng);
                hp1.insertTrackHit(p << 3);
                hp2.insertTrackHit(p << 3);
            } else {
                hp1.insertTrackHit(ugen(eng) << 3);
                hp2.insertTrackHit(ugen(eng) << 3);
            }
        }
    
        for (int i = 0; i != 15; ++i) {
            printf("%d,%d ", hp1.getHitPattern(HitPattern::TRACK_HITS, i) >> 3,
                   hp2.getHitPattern(HitPattern::TRACK_HITS, i) >> 3);
        }
        printf("\n");
    
        PatternSet<15> p1(HitPattern::TRACK_HITS, hp1), p2(HitPattern::TRACK_HITS, hp2);
    
        PatternSet<15> comm = commonHits(p1, p2);
        std::cout << "common " << comm.size() << std::endl;
        for (auto p : comm) {
            printf("%d ", int(p));
        }
        printf("\n");
    
        assert(p1.size() == 15);
        assert(p2.size() == 15);
        for (int i = 0; i != 14; ++i) {
            printf("%d,%d ", int(p1[i]), int(p2[i]));
            assert(p1[i] != 0);
            assert(p2[i] != 0);
            assert(p1[i] <= p1[i + 1]);
            assert(p2[i] <= p2[i + 1]);
        }
        printf("\n");
        return 0;
    }
  }
}

int main() {
  return test::TestHitPattern::test();
}
