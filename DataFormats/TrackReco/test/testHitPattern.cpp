#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include <random>
#include <algorithm>
#include <cstdio>
#include <iostream>

namespace test {
  namespace TestHitPattern {
    using namespace reco;

    int test() {
      {
        // Phase0, only for testing
        TrackerTopology::PixelBarrelValues ttopo_pxb{
            16,
            8,
            2,
            0,
            0xF,
            0xFF,
            0x3F,
            0x0};  //DoubleMask is not used in Phase0, so initializing the corresponding starting bit and mask to 0
        TrackerTopology::PixelEndcapValues ttopo_pxf{23, 16, 10, 8, 2, 0x3, 0xF, 0x3F, 0x3, 0x3F};
        TrackerTopology::TECValues ttopo_tec{18, 14, 12, 8, 5, 2, 0, 0x3, 0xF, 0x3, 0xF, 0x7, 0x7, 0x3};
        TrackerTopology::TIBValues ttopo_tib{14, 12, 10, 4, 2, 0, 0x7, 0x3, 0x3, 0x7F, 0x3, 0x3};
        TrackerTopology::TIDValues ttopo_tid{13, 11, 9, 7, 2, 0, 0x3, 0x3, 0x3, 0x3, 0x1f, 0x3};
        TrackerTopology::TOBValues ttopo_tob{14, 12, 5, 2, 0, 0x7, 0x3, 0x7f, 0x7, 0x3};
        TrackerTopology ttopo{ttopo_pxb, ttopo_pxf, ttopo_tec, ttopo_tib, ttopo_tid, ttopo_tob};

        const uint32_t radial_detids[] = {
            402666125,  //TID r1
            402668833,  //TID r2
            402673476,  //TID r3
            470066725,  //TEC r1
            470390853,  //TEC r2
            470114664,  //TEC r3
            470131344,  //TEC r4
            470079661,  //TEC r5
            470049476,  //TEC r6
            470045428,  //TEC r7
            0x628f3a3c  // MTD BTL
        };

        HitPattern hp;
        auto i = 0;
        for (auto id : radial_detids) {
          hp.appendHit(id, (i++ == 1) ? TrackingRecHit::missing : TrackingRecHit::valid, ttopo);
        }
        hp.appendHit(radial_detids[2], TrackingRecHit::missing, ttopo);
        hp.appendHit(radial_detids[8], TrackingRecHit::missing, ttopo);

        std::cout << hp.numberOfValidTrackerHits() << ' ' << hp.numberOfValidPixelHits() << ' '
                  << hp.numberOfValidStripHits() << ' ' << hp.numberOfValidTimingHits() << std::endl;
        std::cout << hp.pixelLayersWithMeasurement() << ' ' << hp.stripLayersWithMeasurement() << std::endl;
        std::cout << hp.numberOfValidStripLayersWithMonoAndStereo() << std::endl;
        std::cout << hp.pixelLayersWithoutMeasurement(HitPattern::TRACK_HITS) << ' '
                  << hp.stripLayersWithoutMeasurement(HitPattern::TRACK_HITS) << std::endl;

        assert(hp.numberOfValidTrackerHits() == 9);
        assert(hp.numberOfValidPixelHits() == 0);
        assert(hp.numberOfValidStripHits() == 9);
        assert(hp.pixelLayersWithMeasurement() == 0);
        assert(hp.stripLayersWithMeasurement() == 7);
        assert(hp.numberOfValidStripLayersWithMonoAndStereo() == 1);
        assert(hp.pixelLayersWithoutMeasurement(HitPattern::TRACK_HITS) == 0);
        assert(hp.stripLayersWithoutMeasurement(HitPattern::TRACK_HITS) == 1);
        assert(hp.numberOfValidTimingHits() == 1);
      }

      {
        uint16_t oldHitPattern[50] = {20113, 44149, 2321, 19529, 37506, 34993, 11429, 12644, 23051, 13124, 26, 0};

        uint8_t hitCount = 15;
        uint8_t beginTrackHits = 3;
        uint8_t endTrackHits = 15;
        uint8_t beginInner = 0;
        uint8_t endInner = 0;
        uint8_t beginOuter = 0;
        uint8_t endOuter = 3;

        HitPattern hp;
        HitPattern::fillNewHitPatternWithOldHitPattern_v12(
            oldHitPattern, hitCount, beginTrackHits, endTrackHits, beginInner, endInner, beginOuter, endOuter, &hp);

        assert(hp.numberOfValidTrackerHits() == 12);
        assert(hp.numberOfValidPixelHits() == 4);
        assert(hp.numberOfValidPixelBarrelHits() == 4);
        assert(hp.numberOfValidPixelEndcapHits() == 0);
        assert(hp.numberOfValidStripHits() == 8);
        assert(hp.numberOfValidStripTIBHits() == 6);
        assert(hp.numberOfValidStripTIDHits() == 0);
        assert(hp.numberOfValidStripTOBHits() == 2);
        assert(hp.numberOfValidStripTECHits() == 0);

        assert(hp.numberOfTimingHits() == 0);
        assert(hp.numberOfValidTimingHits() == 0);
        assert(hp.numberOfValidTimingBTLHits() == 0);
        assert(hp.numberOfValidTimingETLHits() == 0);

        assert(hp.numberOfLostTimingHits() == 0);
        assert(hp.numberOfLostTimingBTLHits() == 0);
        assert(hp.numberOfLostTimingETLHits() == 0);

        assert(hp.numberOfMuonHits() == 0);
        assert(hp.numberOfValidMuonHits() == 0);
        assert(hp.numberOfValidMuonDTHits() == 0);
        assert(hp.numberOfValidMuonCSCHits() == 0);  //20
        assert(hp.numberOfValidMuonRPCHits() == 0);
        assert(hp.numberOfValidMuonGEMHits() == 0);
        assert(hp.numberOfValidMuonME0Hits() == 0);

        assert(hp.numberOfLostMuonHits() == 0);
        assert(hp.numberOfLostMuonDTHits() == 0);
        assert(hp.numberOfLostMuonCSCHits() == 0);
        assert(hp.numberOfLostMuonRPCHits() == 0);
        assert(hp.numberOfLostMuonGEMHits() == 0);
        assert(hp.numberOfLostMuonME0Hits() == 0);

        assert(hp.numberOfBadHits() == 0);  // 30
        assert(hp.numberOfBadMuonHits() == 0);
        assert(hp.numberOfBadMuonDTHits() == 0);
        assert(hp.numberOfBadMuonCSCHits() == 0);
        assert(hp.numberOfBadMuonRPCHits() == 0);
        assert(hp.numberOfBadMuonGEMHits() == 0);
        assert(hp.numberOfBadMuonME0Hits() == 0);

        assert(hp.numberOfInactiveHits() == 0);
        assert(hp.numberOfInactiveTrackerHits() == 0);
        //assert(hp.numberOfInactiveTimingHits() );

        assert(hp.numberOfValidStripLayersWithMonoAndStereo() == 3);

        assert(hp.trackerLayersWithMeasurementOld() == 9);  //40
        assert(hp.trackerLayersWithMeasurement() == 9);
        assert(hp.pixelLayersWithMeasurementOld() == 4);
        assert(hp.pixelLayersWithMeasurement() == 4);
        assert(hp.stripLayersWithMeasurement() == 5);
        assert(hp.pixelBarrelLayersWithMeasurement() == 4);
        assert(hp.pixelEndcapLayersWithMeasurement() == 0);
        assert(hp.stripTIBLayersWithMeasurement() == 4);
        assert(hp.stripTIDLayersWithMeasurement() == 0);
        assert(hp.stripTOBLayersWithMeasurement() == 1);
        assert(hp.stripTECLayersWithMeasurement() == 0);  //50

        assert(hp.trackerLayersNull() == 20);
        assert(hp.pixelLayersNull() == 3);
        assert(hp.stripLayersNull() == 17);
        assert(hp.pixelBarrelLayersNull() == 0);
        assert(hp.pixelEndcapLayersNull() == 3);
        assert(hp.stripTIBLayersNull() == 0);
        assert(hp.stripTIDLayersNull() == 3);
        assert(hp.stripTOBLayersNull() == 5);
        assert(hp.stripTECLayersNull() == 9);
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
          hp1.insertTrackHit((123 << 3) + 1);  // invalid
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
        printf("%d,%d ",
               hp1.getHitPattern(HitPattern::TRACK_HITS, i) >> 3,
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
  }  // namespace TestHitPattern
}  // namespace test

int main() { return test::TestHitPattern::test(); }
