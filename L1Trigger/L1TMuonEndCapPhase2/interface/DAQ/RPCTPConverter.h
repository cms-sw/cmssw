#ifndef L1Trigger_L1TMuonEndCapPhase2_RPCTPConverter_h
#define L1Trigger_L1TMuonEndCapPhase2_RPCTPConverter_h

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFfwd.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFTypes.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/TPrimitives.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/TPConverters.h"

namespace emtf::phase2 {

    class RPCTPConverter: public TPConverter {
        public:
            explicit RPCTPConverter(const EMTFContext&,
                    const int&, const int&);

            ~RPCTPConverter();

            void convert(
                    const TriggerPrimitive&,
                    const TPInfo&,
                    EMTFHit&) const final;

        private:
            const EMTFContext& context_;

            int endcap_, sector_;
    };

}  // namespace emtf::phase2

#endif  // L1Trigger_L1TMuonEndCapPhase2_RPCTPConverter_h
