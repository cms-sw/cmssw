#ifndef L1Trigger_L1TMuonEndCapPhase2_CSCTPConverter_h
#define L1Trigger_L1TMuonEndCapPhase2_CSCTPConverter_h

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFfwd.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFTypes.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/TPrimitives.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/TPConverters.h"

namespace emtf::phase2 {

    class CSCTPConverter: public TPConverter {
        public:
            explicit CSCTPConverter(const EMTFContext&,
                    const int&, const int&);

            ~CSCTPConverter();

            void convert(
                    const TriggerPrimitive&,
                    const TPInfo&,
                    EMTFHit&) const final;

        private:
            const EMTFContext& context_;

            int endcap_, sector_;
    };  // namespace emtf::phase2
}

#endif  // L1Trigger_L1TMuonEndCapPhase2_CSCTPConverter_h
