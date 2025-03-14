#ifndef RecoTracker_LSTCore_interface_alpaka_LST_h
#define RecoTracker_LSTCore_interface_alpaka_LST_h

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/LSTESData.h"
#include "RecoTracker/LSTCore/interface/alpaka/LSTInputDeviceCollection.h"

#include <cstdlib>
#include <numeric>
#include <alpaka/alpaka.hpp>

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  class LSTEvent;

  class LST {
  public:
    LST() = default;

    void run(Queue& queue,
             bool verbose,
             const float ptCut,
             LSTESData<Device> const* deviceESData,
             LSTInputDeviceCollection const* lstInputDC,
             bool no_pls_dupclean,
             bool tc_pls_triplets);
    std::vector<std::vector<unsigned int>> const& hits() const { return out_tc_hitIdxs_; }
    std::vector<unsigned int> const& len() const { return out_tc_len_; }
    std::vector<int> const& seedIdx() const { return out_tc_seedIdx_; }
    std::vector<short> const& trackCandidateType() const { return out_tc_trackCandidateType_; }

  private:
    void getOutput(LSTEvent& event);

    // Output vectors
    std::vector<std::vector<unsigned int>> out_tc_hitIdxs_;
    std::vector<unsigned int> out_tc_len_;
    std::vector<int> out_tc_seedIdx_;
    std::vector<short> out_tc_trackCandidateType_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif
