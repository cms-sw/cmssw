#ifndef L1Trigger_Phase2L1GT_L1GTChannelMapping_h
#define L1Trigger_Phase2L1GT_L1GTChannelMapping_h

#include "L1Trigger/DemonstratorTools/interface/BoardDataWriter.h"
#include "L1Trigger/DemonstratorTools/interface/utilities.h"

#include <array>
#include <vector>
#include <tuple>

namespace l1t {

  template <typename T, std::size_t low, std::size_t high, std::size_t incr = 1>
  static constexpr std::array<T, high - low> arange() {
    std::array<T, high - low> array;
    T value = low;
    for (T& el : array) {
      el = value;
      value += incr;
    }
    return array;
  }

  template <typename T, std::size_t low, std::size_t high, std::size_t incr = 1>
  static std::vector<T> vrange() {
    std::array<T, high - low> arr(arange<T, low, high, incr>());
    return std::vector(std::begin(arr), std::end(arr));
  }

  static const l1t::demo::BoardDataWriter::ChannelMap_t INPUT_CHANNEL_MAP_VU9P{
      {{"GTT", 1}, {{6, 0}, vrange<std::size_t, 0, 6>()}},
      {{"GTT", 2}, {{6, 0}, vrange<std::size_t, 6, 12>()}},
      {{"CL2", 1}, {{6, 0}, vrange<std::size_t, 28, 34>()}},
      {{"CL2", 2}, {{6, 0}, vrange<std::size_t, 34, 40>()}},
      {{"GCT", 1}, {{6, 0}, vrange<std::size_t, 54, 60>()}},
      {{"GMT", 1}, {{18, 0}, vrange<std::size_t, 60, 78>()}},
      {{"CL2", 3}, {{6, 0}, vrange<std::size_t, 80, 86>()}},
      {{"GTT", 3}, {{6, 0}, vrange<std::size_t, 104, 110>()}},
      {{"GTT", 4}, {{6, 0}, vrange<std::size_t, 110, 116>()}}};

  static const l1t::demo::BoardDataWriter::ChannelMap_t INPUT_CHANNEL_MAP_VU13P{
      {{"GTT", 1}, {{6, 0}, vrange<std::size_t, 0, 6>()}},
      {{"GTT", 2}, {{6, 0}, vrange<std::size_t, 6, 12>()}},
      {{"GCT", 1}, {{6, 0}, vrange<std::size_t, 24, 30>()}},
      {{"CL2", 1}, {{6, 0}, vrange<std::size_t, 32, 38>()}},
      {{"CL2", 2}, {{6, 0}, vrange<std::size_t, 38, 44>()}},
      {{"GMT", 1}, {{18, 0}, {48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 68, 69, 70, 71, 72, 73}}},
      {{"CL2", 3}, {{6, 0}, vrange<std::size_t, 80, 86>()}},
      {{"GTT", 3}, {{6, 0}, vrange<std::size_t, 112, 118>()}},
      {{"GTT", 4}, {{6, 0}, vrange<std::size_t, 118, 124>()}}};

  typedef std::array<std::tuple<const char*, std::size_t, std::size_t>, 27> GTOutputChannelMap_t;

  static constexpr GTOutputChannelMap_t OUTPUT_CHANNELS_VU9P{{{"GTTPromptJets", 2, 6},
                                                              {"GTTDisplacedJets", 6, 10},
                                                              {"GTTPromptHtSum", 10, 11},
                                                              {"GTTDisplacedHtSum", 11, 12},
                                                              {"GTTEtSum", 12, 13},
                                                              {"GTTHadronicTaus", 13, 16},
                                                              {"CL2JetsSC4", 24, 28},
                                                              {"CL2JetsSC8", 28, 32},
                                                              {"CL2Taus", 34, 37},
                                                              {"CL2HtSum", 37, 38},
                                                              {"CL2EtSum", 38, 39},
                                                              {"GCTNonIsoEg", 48, 50},
                                                              {"GCTIsoEg", 50, 52},
                                                              {"GCTJets", 52, 54},
                                                              {"GCTTaus", 54, 56},
                                                              {"GCTHtSum", 56, 57},
                                                              {"GCTEtSum", 57, 58},
                                                              {"GMTSaPromptMuons", 60, 62},
                                                              {"GMTSaDisplacedMuons", 62, 64},
                                                              {"GMTTkMuons", 64, 67},
                                                              {"GMTTopo", 67, 69},
                                                              {"CL2Electrons", 80, 83},
                                                              {"CL2Photons", 83, 86},
                                                              {"GTTPhiCandidates", 104, 107},
                                                              {"GTTRhoCandidates", 107, 110},
                                                              {"GTTBsCandidates", 110, 113},
                                                              {"GTTPrimaryVert", 113, 115}}};

  static constexpr GTOutputChannelMap_t OUTPUT_CHANNELS_VU13P{{{"GTTPromptJets", 2, 6},
                                                               {"GTTDisplacedJets", 6, 10},
                                                               {"GTTPromptHtSum", 10, 11},
                                                               {"GTTDisplacedHtSum", 11, 12},
                                                               {"GTTEtSum", 12, 13},
                                                               {"GTTHadronicTaus", 13, 16},
                                                               {"GCTNonIsoEg", 26, 28},
                                                               {"GCTIsoEg", 28, 30},
                                                               {"GCTJets", 30, 32},
                                                               {"CL2JetsSC4", 32, 36},
                                                               {"CL2JetsSC8", 36, 40},
                                                               {"CL2Taus", 40, 43},
                                                               {"CL2HtSum", 43, 44},
                                                               {"CL2EtSum", 44, 45},
                                                               {"GMTSaPromptMuons", 68, 70},
                                                               {"GMTSaDisplacedMuons", 70, 72},
                                                               {"GMTTkMuons", 72, 75},
                                                               {"GMTTopo", 75, 77},
                                                               {"CL2Electrons", 80, 83},
                                                               {"CL2Photons", 83, 86},
                                                               {"GCTTaus", 96, 98},
                                                               {"GCTHtSum", 98, 99},
                                                               {"GCTEtSum", 99, 100},
                                                               {"GTTPhiCandidates", 112, 115},
                                                               {"GTTRhoCandidates", 115, 118},
                                                               {"GTTBsCandidates", 118, 121},
                                                               {"GTTPrimaryVert", 121, 123}}};

}  // namespace l1t

#endif  // L1Trigger_Phase2L1GT_L1GTChannelMapping_h
