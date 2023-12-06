#ifndef L1Trigger_L1TMuonEndCapPhase2_RoadSortingLayer_h
#define L1Trigger_L1TMuonEndCapPhase2_RoadSortingLayer_h

#include <array>
#include <vector>

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFfwd.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFTypes.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFConstants.h"

namespace emtf::phase2::algo {

    class RoadSortingLayer {
        public:
            RoadSortingLayer(const EMTFContext&);

            ~RoadSortingLayer();

            void apply(
                const int&,
                const std::vector<road_collection_t>&,
                std::vector<road_t>&
            ) const;
        private:
            const EMTFContext& context_;
    };

}

#endif  // L1Trigger_L1TMuonEndCapPhase2_RoadSortingLayer_h not defined
