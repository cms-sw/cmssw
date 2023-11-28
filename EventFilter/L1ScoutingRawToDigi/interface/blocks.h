#ifndef L1ScoutingRawToDigi_blocks_h
#define L1ScoutingRawToDigi_blocks_h

#include <cstdint>

namespace l1ScoutingRun3 {

  namespace ugmt {

    struct muon {
      uint32_t f;
      uint32_t s;
      uint32_t extra;
    };

    struct block {
      uint32_t bx;
      uint32_t orbit;
      muon mu[16];
    };
  }  // namespace ugmt

  namespace demux {

    // unrolled frame block
    struct block {
      uint32_t header;
      uint32_t bx;
      uint32_t orbit;
      uint32_t link0;
      uint32_t jet2[6];
      uint32_t link1;
      uint32_t jet1[6];
      uint32_t link2;
      uint32_t egamma2[6];
      uint32_t link3;
      uint32_t egamma1[6];
      uint32_t link4;
      uint32_t empty[6];
      uint32_t link5;
      uint32_t sum[6];
      uint32_t link6;
      uint32_t tau2[6];
      uint32_t link7;
      uint32_t tau1[6];
    };
  }  // namespace demux

}  // namespace l1ScoutingRun3
#endif  // L1ScoutingRawToDigi_blocks_h
