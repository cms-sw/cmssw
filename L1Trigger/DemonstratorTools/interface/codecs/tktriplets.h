#ifndef L1Trigger_DemonstratorTools_codecs_tktriplets_h
#define L1Trigger_DemonstratorTools_codecs_tktriplets_h

#include <array>
#include <vector>

#include "ap_int.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/L1TCorrelator/interface/TkTriplet.h"
#include "DataFormats/L1TCorrelator/interface/TkTripletFwd.h"
#include "DataFormats/L1Trigger/interface/TkTripletWord.h"
#include "L1Trigger/L1TTrackMatch/interface/TkTripletEmuAlgo.h"
#include "L1Trigger/DemonstratorTools/interface/codecs/tracks.h"

namespace l1t::demo::codecs {

  ap_uint<128> encodeTriplet(const l1t::TkTripletWord& t);

  void encodeTripletLinks(std::array<std::vector<ap_uint<128>>, 1>& tripletWords,
                          std::array<std::vector<ap_uint<64>>, 1>& linkData);

  // Encodes TkTriplet collection onto 1 'logical' output link
  std::array<std::vector<ap_uint<64>>, 1> encodeTriplets(const edm::View<l1t::TkTripletWord>&);

}  // namespace l1t::demo::codecs

#endif
