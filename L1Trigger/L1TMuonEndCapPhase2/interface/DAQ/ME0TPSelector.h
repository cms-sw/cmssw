#ifndef L1Trigger_L1TMuonEndCapPhase2_ME0TPSelector_h
#define L1Trigger_L1TMuonEndCapPhase2_ME0TPSelector_h

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFfwd.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFTypes.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/TPrimitives.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/TPSelectors.h"

// 6 in GE1/1; 3 in GE2/1; 3 in ME0; 3 from neighbor sector. (unconfirmed!)
#define NUM_GEM_CHAMBERS 15

namespace emtf::phase2 {

    class ME0TPSelector: public TPSelector {
        public:
            explicit ME0TPSelector(
                    const EMTFContext&,
                    const int&, const int&
            );

            ~ME0TPSelector();

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
                    const TPSelection&
            ) const;
    };

}  // namespace emtf::phase2

#endif  // L1Trigger_L1TMuonEndCapPhase2_ME0TPSelector_h
