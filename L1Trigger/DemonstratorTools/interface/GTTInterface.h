#ifndef L1Trigger_DemonstratorTools_GTTInterface_h
#define L1Trigger_DemonstratorTools_GTTInterface_h

#include <cstddef>

#include "L1Trigger/DemonstratorTools/interface/LinkId.h"
#include "L1Trigger/DemonstratorTools/interface/ChannelSpec.h"
#include "L1Trigger/DemonstratorTools/interface/EventData.h"
#include "L1Trigger/DemonstratorTools/interface/FileFormat.h"
#include "L1Trigger/DemonstratorTools/interface/Frame.h"

namespace l1t::demo {
  //Global Trigger Interface Doc - https://www.overleaf.com/read/qqfdtgcybhvh#746997
  //Link 1 of GTT payload to GT
  /* size_t nJetToWords ( constexpr size_t n ) { if ( n % 2 == 0 ) return n * 3 / 2; else throw an exception} */
  /* size_t nSumToWords ( constexpr size_t n ) { return n; } */
  /* size_t nTauToWords ( constexpr size_t n ) { return n * 3 / 2; } */
  /* size_t nJetToWords ( constexpr size_t nJet ) { return nJet * 3 / 2; } */
  /* namespace gttToGT1 { */
  /*   //12 prompt and 12 displaced jets with 2 64-bit words each */
  /*   //1 Prompt HT Miss, 1 Displaced HT Miss, 1 ET Miss with 1 64-bit word each */
  /*   static constexpr size_t nPromptJets = 12; */
  /*   static constexpr size_t nDisplacedJets = 12; */
  /*   static constexpr size_t nPromptHTMiss = 1; */
  /*   static constexpr size_t nDisplacedHTMiss = 1; */
  /*   static constexpr size_t nETMiss = 1; */
  /*   static constexpr size_t kPromptJetStart = 0; */
  /*   static constexpr size_t kPromptJetEnd = kPromptJetStart + nPromptJets - 1; */
  /*   static constexpr size_t kDisplacedJetStart = kPromptJetEnd + 1; */
  /*   static constexpr size_t kDisplacedJetEnd = kDisplacedJetStart + 24 - 1; */
  /*   static constexpr size_t kPromptHTMissStart = kDisplacedJetEnd + 1; */
  /*   static constexpr size_t kPromptHTMissEnd = kPromptHTMissStart + 1 - 1; */
  /*   static constexpr size_t kDisplacedHTMissStart = kPromptHTMissEnd + 1; */
  /*   static constexpr size_t kDisplacedHTMissEnd = kDisplacedHTMissStart + 1 - 1; */
  /*   static constexpr size_t kETMissStart = kDisplacedHTMissEnd + 1; */
  /*   static constexpr size_t kETMissEnd = kETMissStart + 1 - 1; */
  /*   static constexpr size_t kEmptyStart = kETMissEnd + 1; */
  /*   static constexpr size_t kEmptyEnd = 53; */
  /* } */
  /* namespace gttToGT2 { */
  /*   //12 taus with 96-bit words, or 18 64-bit words */
  /*   static constexpr size_t kHadronicTauStart = 0; */
  /*   static constexpr size_t kHadronicTauEnd = kHadronicTauStart + 18 - 1; */
  /*   static constexpr size_t kEmptyStart = kHadronicTauEnd + 1; */
  /*   static constexpr size_t kEmptyEnd = 53; */
  /* } */
  /* namespace gttToGT3 { */
  /*   //12 phi meson candidates and 12 rho meson candidates with 1.5 64-bit words each */
  /*   //2 B_s candidates with 1 64-bit word each */
  /*   static constexpr size_t kPhiMesonStart = 0; */
  /*   static constexpr size_t kPhiMesonEnd = kPhiMesonStart + 18 - 1; */
  /*   static constexpr size_t kRhoMesonStart = kPhiMesonEnd + 1; */
  /*   static constexpr size_t kRhoMesonEnd = kRhoMesonStart + 18 - 1; */
  /*   static constexpr size_t kBsStart = kRhoMesonEnd + 1; */
  /*   static constexpr size_t kBsEnd = kBsStart + 2 - 1; */
  /*   static constexpr size_t kEmptyStart = kRhoMesonEnd + 1; */
  /*   static constexpr size_t kEmptyEnd = 53; */
  /* } */
  /* namespace gttToGT4 { */
  /*   //12 prompt isolated tracks and 12 displaced isolated tracks with 1.5 64-bit words each */
  /*   //10 prompt vertices and 2 displaced vertices with 1 64-bit word each */
  /*   static constexpr size_t kPromptIsoTracksStart = 0; */
  /*   static constexpr size_t kPromptIsoTracksEnd = kPromptIsoTracksStart + 18 - 1; */
  /*   static constexpr size_t kDisplacedIsoTracksStart = kPromptIsoTracksEnd + 1; */
  /*   static constexpr size_t kDisplacedIsoTracksEnd = kDisplacedIsoTracksStart + 18 - 1; */
  /*   static constexpr size_t kPromptVerticesStart = kDisplacedIsoTracksEnd + 1; */
  /*   static constexpr size_t kPromptVerticesEnd = kPromptVerticesStart + 10 - 1; */
  /*   static constexpr size_t kDisplacedVerticesStart = kPromptVerticesEnd + 1; */
  /*   static constexpr size_t kDisplacedVerticesEnd = kDisplacedVerticesStart + 2 - 1; */
  /*   static constexpr size_t kEmptyStart = kDisplacedVerticesEnd + 1; */
  /*   static constexpr size_t kEmptyEnd = 53; */
  /* } */
  
  // map of logical channel ID -> [TMUX period, interpacket-gap & offset; channel indices]
  typedef std::map<LinkId, std::pair<ChannelSpec, std::vector<size_t>>> ChannelMap_t;
  
  static constexpr size_t kFramesPerTMUXPeriod = 9;
  static constexpr size_t kGapLengthInput = 6; //defined in terms of nTracks * (3/2) - 3 * 54?
  static constexpr size_t kGapLengthOutputToCorrelator = 44; //can be defined in terms of 54 - nVertices?
  static constexpr size_t kGapLengthOutputToGlobalTriggerSums = 3;
  static constexpr size_t kGapLengthOutputToGlobalTriggerTaus = 36;
  static constexpr size_t kGapLengthOutputToGlobalTriggerMesons = 15;
  static constexpr size_t kGapLengthOutputToGlobalTriggerVertices = 6;
  static constexpr size_t kTrackTMUX = 18; //TMUX of the TrackFindingProcessors
  static constexpr size_t kGTTBoardTMUX = 6; //TMUX of the GTT in the current configuration: 6 boards running 3 events in parallel, with a paired board running parallel algorithms
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
  //This is actually a ChannelMap_t... we only need the actual map below
  /* static const std::map<l1t::demo::LinkId, std::pair<l1t::demo::ChannelSpec, std::vector<size_t>>> */
  /*     kChannelSpecsOutputToCorrelator = { */
  /* static const ChannelMap_t kChannelSpecsOutputToCorrelator = { */
          /* logical channel within time slice -> {{link TMUX, inter-packet gap}, vector of channel indices} */
          /* {{"vertices", 0}, {{kGTTBoardTMUX, kGapLengthOutputToCorrelator}, {0}}}}; */

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
