#ifndef L1Trigger_DemonstratorTools_GTTInterface_h
#define L1Trigger_DemonstratorTools_GTTInterface_h

#include <cstddef>

#include "L1Trigger/DemonstratorTools/interface/LinkId.h"
#include "L1Trigger/DemonstratorTools/interface/ChannelSpec.h"
#include "L1Trigger/DemonstratorTools/interface/EventData.h"
#include "L1Trigger/DemonstratorTools/interface/FileFormat.h"
#include "L1Trigger/DemonstratorTools/interface/Frame.h"

namespace l1t::demo {
  // map of logical channel ID -> [TMUX period, interpacket-gap & offset; channel indices]
  typedef std::map<LinkId, std::pair<ChannelSpec, std::vector<size_t>>> ChannelMap_t;

  static constexpr size_t kFramesPerTMUXPeriod = 9;
  static constexpr size_t kGapLengthInput = 6;                //defined in terms of nTracks * (3/2) - 3 * 54?
  static constexpr size_t kGapLengthOutputToCorrelator = 44;  //can be defined in terms of 54 - nVertices?
  static constexpr size_t kGapLengthOutputToGlobalTriggerSums = 3;
  static constexpr size_t kGapLengthOutputToGlobalTriggerTaus = 36;
  static constexpr size_t kGapLengthOutputToGlobalTriggerMesons = 15;
  static constexpr size_t kGapLengthOutputToGlobalTriggerVertices = 6;
  static constexpr size_t kTrackTMUX = 18;  //TMUX of the TrackFindingProcessors
  static constexpr size_t kGTTBoardTMUX =
      6;  //TMUX of the GTT in the current configuration: 6 boards running 3 events in parallel, with a paired board running parallel algorithms
  static constexpr size_t kMaxLinesPerFile = 1024;

  static constexpr size_t kVertexChanIndex = 0;

  // TRACKS from TFP
  static const std::map<l1t::demo::LinkId, std::vector<size_t>> kChannelIdsInput = {
      /* logical channel within time slice -> vector of channel indices (one entry per time slice) */
      /* for first link in a time slice, the channel index is 1 for 1st time slice, channel 19 in the 2nd*/
      {{"tracks", 0}, {0, 18, 36}},
      {{"tracks", 1}, {1, 19, 37}},
      {{"tracks", 2}, {2, 20, 38}},
      {{"tracks", 3}, {3, 21, 39}},
      {{"tracks", 4}, {4, 22, 40}},
      {{"tracks", 5}, {5, 23, 41}},
      {{"tracks", 6}, {6, 24, 42}},
      {{"tracks", 7}, {7, 25, 43}},
      {{"tracks", 8}, {8, 26, 44}},
      {{"tracks", 9}, {9, 27, 45}},
      {{"tracks", 10}, {10, 28, 46}},
      {{"tracks", 11}, {11, 29, 47}},
      {{"tracks", 12}, {12, 30, 48}},
      {{"tracks", 13}, {13, 31, 49}},
      {{"tracks", 14}, {14, 32, 50}},
      {{"tracks", 15}, {15, 33, 51}},
      {{"tracks", 16}, {16, 34, 52}},
      {{"tracks", 17}, {17, 35, 53}}};

  static const ChannelMap_t kChannelMapInput = {
      /* logical channel within time slice -> {{link TMUX, inter-packet gap}, vector of channel indices} */
      {{"tracks", 0}, {{kTrackTMUX, kGapLengthInput}, {0, 18, 36}}},
      {{"tracks", 1}, {{kTrackTMUX, kGapLengthInput}, {1, 19, 37}}},
      {{"tracks", 2}, {{kTrackTMUX, kGapLengthInput}, {2, 20, 38}}},
      {{"tracks", 3}, {{kTrackTMUX, kGapLengthInput}, {3, 21, 39}}},
      {{"tracks", 4}, {{kTrackTMUX, kGapLengthInput}, {4, 22, 40}}},
      {{"tracks", 5}, {{kTrackTMUX, kGapLengthInput}, {5, 23, 41}}},
      {{"tracks", 6}, {{kTrackTMUX, kGapLengthInput}, {6, 24, 42}}},
      {{"tracks", 7}, {{kTrackTMUX, kGapLengthInput}, {7, 25, 43}}},
      {{"tracks", 8}, {{kTrackTMUX, kGapLengthInput}, {8, 26, 44}}},
      {{"tracks", 9}, {{kTrackTMUX, kGapLengthInput}, {9, 27, 45}}},
      {{"tracks", 10}, {{kTrackTMUX, kGapLengthInput}, {10, 28, 46}}},
      {{"tracks", 11}, {{kTrackTMUX, kGapLengthInput}, {11, 29, 47}}},
      {{"tracks", 12}, {{kTrackTMUX, kGapLengthInput}, {12, 30, 48}}},
      {{"tracks", 13}, {{kTrackTMUX, kGapLengthInput}, {13, 31, 49}}},
      {{"tracks", 14}, {{kTrackTMUX, kGapLengthInput}, {14, 32, 50}}},
      {{"tracks", 15}, {{kTrackTMUX, kGapLengthInput}, {15, 33, 51}}},
      {{"tracks", 16}, {{kTrackTMUX, kGapLengthInput}, {16, 34, 52}}},
      {{"tracks", 17}, {{kTrackTMUX, kGapLengthInput}, {17, 35, 53}}}};

  static const std::map<std::string, l1t::demo::ChannelSpec> kChannelSpecsInput = {
      /* interface name -> {link TMUX, inter-packet gap} */
      {"tracks", {kTrackTMUX, kGapLengthInput}}};

  //OUTPUTS to Correlator
  static const ChannelMap_t kChannelMapOutputToCorrelator = {
      /* logical channel within time slice -> {{link TMUX, inter-packet gap}, vector of channel indices} */
      {{"vertices", 0}, {{kGTTBoardTMUX, kGapLengthOutputToCorrelator}, {0}}}};

  //OUTPUTS to Global Trigger
  static const std::map<l1t::demo::LinkId, std::vector<size_t>> kChannelIdsOutputToGlobalTrigger = {
      /* logical channel within time slice -> vector of channel indices (one entry per time slice) */
      {{"sums", 0}, {0}},
      {{"taus", 1}, {1}},
      {{"mesons", 2}, {2}},
      {{"vertices", 3}, {3}}};

  static const std::map<std::string, l1t::demo::ChannelSpec> kChannelSpecsOutputToGlobalTrigger = {
      /* interface name -> {link TMUX, inter-packet gap} */
      {"sums", {kGTTBoardTMUX, kGapLengthOutputToGlobalTriggerSums}},
      {"taus", {kGTTBoardTMUX, kGapLengthOutputToGlobalTriggerTaus}},
      {"mesons", {kGTTBoardTMUX, kGapLengthOutputToGlobalTriggerMesons}},
      {"vertices", {kGTTBoardTMUX, kGapLengthOutputToGlobalTriggerVertices}}};

}  // namespace l1t::demo

#endif
