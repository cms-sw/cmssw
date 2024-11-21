#ifndef RecoTracker_LSTCore_interface_alpaka_LST_h
#define RecoTracker_LSTCore_interface_alpaka_LST_h

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/LSTESData.h"

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
             std::vector<float> const& see_px,
             std::vector<float> const& see_py,
             std::vector<float> const& see_pz,
             std::vector<float> const& see_dxy,
             std::vector<float> const& see_dz,
             std::vector<float> const& see_ptErr,
             std::vector<float> const& see_etaErr,
             std::vector<float> const& see_stateTrajGlbX,
             std::vector<float> const& see_stateTrajGlbY,
             std::vector<float> const& see_stateTrajGlbZ,
             std::vector<float> const& see_stateTrajGlbPx,
             std::vector<float> const& see_stateTrajGlbPy,
             std::vector<float> const& see_stateTrajGlbPz,
             std::vector<int> const& see_q,
             std::vector<std::vector<int>> const& see_hitIdx,
             std::vector<unsigned int> const& ph2_detId,
             std::vector<float> const& ph2_x,
             std::vector<float> const& ph2_y,
             std::vector<float> const& ph2_z,
             bool no_pls_dupclean,
             bool tc_pls_triplets);
    std::vector<std::vector<unsigned int>> const& hits() const { return out_tc_hitIdxs_; }
    std::vector<unsigned int> const& len() const { return out_tc_len_; }
    std::vector<int> const& seedIdx() const { return out_tc_seedIdx_; }
    std::vector<short> const& trackCandidateType() const { return out_tc_trackCandidateType_; }

  private:
    void prepareInput(std::vector<float> const& see_px,
                      std::vector<float> const& see_py,
                      std::vector<float> const& see_pz,
                      std::vector<float> const& see_dxy,
                      std::vector<float> const& see_dz,
                      std::vector<float> const& see_ptErr,
                      std::vector<float> const& see_etaErr,
                      std::vector<float> const& see_stateTrajGlbX,
                      std::vector<float> const& see_stateTrajGlbY,
                      std::vector<float> const& see_stateTrajGlbZ,
                      std::vector<float> const& see_stateTrajGlbPx,
                      std::vector<float> const& see_stateTrajGlbPy,
                      std::vector<float> const& see_stateTrajGlbPz,
                      std::vector<int> const& see_q,
                      std::vector<std::vector<int>> const& see_hitIdx,
                      std::vector<unsigned int> const& ph2_detId,
                      std::vector<float> const& ph2_x,
                      std::vector<float> const& ph2_y,
                      std::vector<float> const& ph2_z,
                      const float ptCut);

    void getOutput(LSTEvent& event);

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
    std::vector<PixelType> in_pixelType_vec_;
    std::vector<char> in_isQuad_vec_;
    std::vector<std::vector<unsigned int>> out_tc_hitIdxs_;
    std::vector<unsigned int> out_tc_len_;
    std::vector<int> out_tc_seedIdx_;
    std::vector<short> out_tc_trackCandidateType_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif
