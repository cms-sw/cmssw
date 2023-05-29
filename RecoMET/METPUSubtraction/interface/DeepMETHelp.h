#ifndef RecoMET_METPUSubtraction_DeepMETHelp_h
#define RecoMET_METPUSubtraction_DeepMETHelp_h

#include <unordered_map>
#include <cstdint>

namespace deepmet_helper {
  float scale_and_rm_outlier(float val, float scale);

  static const std::unordered_map<int, int32_t> charge_embedding{{-1, 0}, {0, 1}, {1, 2}};
  static const std::unordered_map<int, int32_t> pdg_id_embedding{
      {-211, 0}, {-13, 1}, {-11, 2}, {0, 3}, {1, 4}, {2, 5}, {11, 6}, {13, 7}, {22, 8}, {130, 9}, {211, 10}};
}  // namespace deepmet_helper

#endif
