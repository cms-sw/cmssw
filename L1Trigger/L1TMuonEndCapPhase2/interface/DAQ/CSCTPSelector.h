#ifndef L1Trigger_L1TMuonEndCapPhase2_CSCTPSelector_h
#define L1Trigger_L1TMuonEndCapPhase2_CSCTPSelector_h

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFfwd.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFTypes.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/TPrimitives.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/TPSelectors.h"

// 18 in ME1; 9x3 in ME2,3,4; 9 from neighbor sector.
// Arranged in FW as 6 stations, 9 chambers per station.
#define NUM_CSC_CHAMBERS 6 * 9

namespace emtf::phase2 {

    class CSCTPSelector: public TPSelector {
        public:
            explicit CSCTPSelector(
                    const EMTFContext&,
                    const int&, const int&
            );

            ~CSCTPSelector();

            void select(
                    const TriggerPrimitive&,
                    TPInfo,
                    ILinkTPCMap&
            ) const final;

        private:
            const EMTFContext& context_;

            int endcap_, sector_;

            int get_input_link(const TriggerPrimitive&, TPInfo&) const;

            int calculate_input_link(
                    const int&, const int&,
                    const int&, const int&,
                    const TPSelection&
            ) const;
    };

}  // namespace emtf::phase2

#endif  // L1Trigger_L1TMuonEndCapPhase2_CSCTPSelector_h
