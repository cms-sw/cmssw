#ifndef L1Trigger_L1TMuonEndCapPhase2_TrackBuildingLayer_h
#define L1Trigger_L1TMuonEndCapPhase2_TrackBuildingLayer_h

#include <array>
#include <vector>

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFfwd.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFTypes.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFConstants.h"

namespace emtf::phase2::algo {

    class TrackBuildingLayer {
        // Static
        private:
            static seg_theta_t calc_theta_median(
                std::vector<seg_theta_t>
            );

        // Members
        public:
            TrackBuildingLayer(const EMTFContext&);

            ~TrackBuildingLayer();

            void apply(
                const segment_collection_t&,
                const std::vector<road_t>&,
                const bool&,
                std::vector<track_t>&
            ) const;

        private:
            const EMTFContext& context_;

            void attach_segments(
                const segment_collection_t&,
                const road_t&,
                const bool&,
                track_t&
            ) const;

    };

}

#endif  // L1Trigger_L1TMuonEndCapPhase2_TrackBuildingLayer_h not defined
