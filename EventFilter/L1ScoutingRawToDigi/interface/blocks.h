#ifndef DataFormats_Scouting_BLOCKS_H
#define DataFormats_Scouting_BLOCKS_H

#include <cstdint>
#include <vector>
#include <string>
#include "scales.h"

namespace scoutingRun3 {

namespace ugmt {
    struct hw_data_block{
        std::vector<int> vorbit;
        std::vector<int> vbx;
        std::vector<int> vinterm;
        std::vector<int> vipt;
        std::vector<int> viptunconstrained;
        std::vector<int> vcharge;
        std::vector<int> viso;
        std::vector<int> vindex;
        std::vector<int> vqual;
        std::vector<int> viphi;
        std::vector<int> viphiext;
        std::vector<int> vieta;
        std::vector<int> vietaext;
        std::vector<int> vidxy;

        unsigned int size() {return vipt.size();}
        bool empty() {return vipt.empty();}
        static const unsigned int ncols = 14;

        std::vector<int> iAt(int i) {
            return {
                vorbit[i],
                vbx[i],
                vinterm[i],
                vipt[i],
                viptunconstrained[i],
                vcharge[i],
                viso[i],
                vindex[i],
                vqual[i],
                viphi[i],
                viphiext[i],
                vieta[i],
                vietaext[i],
                vidxy[i],
            };
        }
    };

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
}



namespace demux {
    struct hw_data_block {
        std::vector<int> vorbit;
        std::vector<int> vbx;
        std::vector<int> vET;
        std::vector<int> vType;
        std::vector<int> vEta;
        std::vector<int> vPhi;
        std::vector<int> vIso;

        unsigned int size() {return vET.size();}
        bool empty() {return vET.empty();}
        static const unsigned int ncols = 13;

    };

    struct block_old {
        uint32_t header;
        uint32_t bx;
        uint32_t orbit;
        uint32_t frame[56];     // +8 for extra word containing link number
    };

    // unrolled frame block
    struct block {
          uint32_t header;
          uint32_t bx;
          uint32_t orbit;
          uint32_t link0;
          uint32_t jet1[6];
          uint32_t link1;
          uint32_t jet2[6];
          uint32_t link2;
          uint32_t egamma1[6];
          uint32_t link3;
          uint32_t egamma2[6];
          uint32_t link4;
          uint32_t empty[6];
          uint32_t link5;
          uint32_t sum[6];
          uint32_t link6;
          uint32_t tau1[6];
          uint32_t link7;
          uint32_t tau2[6];
      };
}



namespace bmtf {
    struct block {
        uint64_t stub[8];
    };
}

}
#endif
