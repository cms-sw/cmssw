#ifndef L1Trigger_L1TMuonEndCapPhase2_RPCTPSelector_h
#define L1Trigger_L1TMuonEndCapPhase2_RPCTPSelector_h

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFfwd.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFTypes.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/TPrimitives.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/TPSelectors.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/RPCUtils.h"

// Arranged in FW as 7 stations, 6 chambers per station.
// For Phase 2, add RE1/3, RE2/3, RE3/1, RE4/1 -> 10 chambers per station
#define NUM_RPC_CHAMBERS 7 * 10

namespace emtf::phase2 {

    class RPCTPSelector: public TPSelector {
        public:
            explicit RPCTPSelector(
                    const EMTFContext&,
                    const int&, const int&
            );

            ~RPCTPSelector();

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

#endif  // L1Trigger_L1TMuonEndCapPhase2_RPCTPSelector_h
