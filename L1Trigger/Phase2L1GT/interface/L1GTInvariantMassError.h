#ifndef L1Trigger_Phase2L1GT_InvariantMassError_h
#define L1Trigger_Phase2L1GT_InvariantMassError_h

#include <vector>

namespace l1t {
  struct InvariantMassError {
    const double absoluteError_ = 0;  // GeV/c^2
    const double relativeError_ = 0;  // GeV/c^2 (could also be calculated)
    const double invariantMass_ = 0;  // GeV/c^2 (calculated without LUT)
  };

  typedef std::vector<InvariantMassError> InvariantMassErrorCollection;
}  // namespace l1t

#endif  // L1Trigger_Phase2L1GT_InvariantMassError_h
