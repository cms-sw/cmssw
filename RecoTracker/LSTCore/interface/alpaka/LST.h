#ifndef LST_H
#define LST_H

#ifdef LST_IS_CMSSW_PACKAGE
#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#include "RecoTracker/LSTCore/interface/alpaka/LSTESData.h"
#else
#include "Constants.h"
#include "LSTESData.h"
#endif

#include <cstdlib>
#include <numeric>
#include <mutex>
#include <alpaka/alpaka.hpp>

namespace SDL {
  template <typename>
  class Event;

  template <typename>
  class LST;

  template <>
  class LST<SDL::Acc> {
  public:
    LST() = default;

    void run(QueueAcc& queue,
             bool verbose,
             const LSTESDeviceData<Dev>* deviceESData,
             const std::vector<float> see_px,
             const std::vector<float> see_py,
             const std::vector<float> see_pz,
             const std::vector<float> see_dxy,
             const std::vector<float> see_dz,
             const std::vector<float> see_ptErr,
             const std::vector<float> see_etaErr,
             const std::vector<float> see_stateTrajGlbX,
             const std::vector<float> see_stateTrajGlbY,
             const std::vector<float> see_stateTrajGlbZ,
             const std::vector<float> see_stateTrajGlbPx,
             const std::vector<float> see_stateTrajGlbPy,
             const std::vector<float> see_stateTrajGlbPz,
             const std::vector<int> see_q,
             const std::vector<std::vector<int>> see_hitIdx,
             const std::vector<unsigned int> ph2_detId,
             const std::vector<float> ph2_x,
             const std::vector<float> ph2_y,
             const std::vector<float> ph2_z);
    std::vector<std::vector<unsigned int>> hits() { return out_tc_hitIdxs_; }
    std::vector<unsigned int> len() { return out_tc_len_; }
    std::vector<int> seedIdx() { return out_tc_seedIdx_; }
    std::vector<short> trackCandidateType() { return out_tc_trackCandidateType_; }

  private:
    void prepareInput(const std::vector<float> see_px,
                      const std::vector<float> see_py,
                      const std::vector<float> see_pz,
                      const std::vector<float> see_dxy,
                      const std::vector<float> see_dz,
                      const std::vector<float> see_ptErr,
                      const std::vector<float> see_etaErr,
                      const std::vector<float> see_stateTrajGlbX,
                      const std::vector<float> see_stateTrajGlbY,
                      const std::vector<float> see_stateTrajGlbZ,
                      const std::vector<float> see_stateTrajGlbPx,
                      const std::vector<float> see_stateTrajGlbPy,
                      const std::vector<float> see_stateTrajGlbPz,
                      const std::vector<int> see_q,
                      const std::vector<std::vector<int>> see_hitIdx,
                      const std::vector<unsigned int> ph2_detId,
                      const std::vector<float> ph2_x,
                      const std::vector<float> ph2_y,
                      const std::vector<float> ph2_z);

    void getOutput(SDL::Event<Acc>& event);
    std::vector<unsigned int> getHitIdxs(const short trackCandidateType,
                                         const unsigned int TCIdx,
                                         const unsigned int* TCHitIndices,
                                         const unsigned int* hitIndices);

    // Input and output vectors
    std::vector<float> in_trkX_;
    std::vector<float> in_trkY_;
    std::vector<float> in_trkZ_;
    std::vector<unsigned int> in_hitId_;
    std::vector<unsigned int> in_hitIdxs_;
    std::vector<unsigned int> in_hitIndices_vec0_;
    std::vector<unsigned int> in_hitIndices_vec1_;
    std::vector<unsigned int> in_hitIndices_vec2_;
    std::vector<unsigned int> in_hitIndices_vec3_;
    std::vector<float> in_deltaPhi_vec_;
    std::vector<float> in_ptIn_vec_;
    std::vector<float> in_ptErr_vec_;
    std::vector<float> in_px_vec_;
    std::vector<float> in_py_vec_;
    std::vector<float> in_pz_vec_;
    std::vector<float> in_eta_vec_;
    std::vector<float> in_etaErr_vec_;
    std::vector<float> in_phi_vec_;
    std::vector<int> in_charge_vec_;
    std::vector<unsigned int> in_seedIdx_vec_;
    std::vector<int> in_superbin_vec_;
    std::vector<int8_t> in_pixelType_vec_;
    std::vector<char> in_isQuad_vec_;
    std::vector<std::vector<unsigned int>> out_tc_hitIdxs_;
    std::vector<unsigned int> out_tc_len_;
    std::vector<int> out_tc_seedIdx_;
    std::vector<short> out_tc_trackCandidateType_;
  };

}  // namespace SDL

#endif
