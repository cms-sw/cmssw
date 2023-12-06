#ifndef L1Trigger_L1TMuonEndCapPhase2_TemplateUtils_h
#define L1Trigger_L1TMuonEndCapPhase2_TemplateUtils_h

#include <array>
#include <vector>

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFfwd.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFTypes.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFConstants.h"

namespace emtf::phase2 {

    template <typename T, typename F>
        T when(const bool& condition, const T& if_true, const F& if_false) {
            return condition ? if_true : static_cast<T>(if_false);
        }

    template <typename T>
        T when(const bool& condition, const T& if_true, const T& if_false) {
            return condition ? if_true : if_false;
        }

}

#endif  // L1Trigger_L1TMuonEndCapPhase2_TemplateUtils_h not defined
