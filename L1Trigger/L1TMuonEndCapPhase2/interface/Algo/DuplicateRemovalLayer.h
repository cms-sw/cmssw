#ifndef L1Trigger_L1TMuonEndCapPhase2_DuplicateRemovalLayer_h
#define L1Trigger_L1TMuonEndCapPhase2_DuplicateRemovalLayer_h

#include <array>
#include <vector>

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFfwd.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFTypes.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFConstants.h"

namespace emtf::phase2::algo {

    class DuplicateRemovalLayer {
        public:
            DuplicateRemovalLayer(const EMTFContext&);

            ~DuplicateRemovalLayer();

            void apply(
                    std::vector<track_t>&
            ) const;

        private:
            const EMTFContext& context_;
    };

}

#endif  // L1Trigger_L1TMuonEndCapPhase2_DuplicateRemovalLayer_h not defined
