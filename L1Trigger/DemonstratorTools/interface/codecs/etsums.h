
#ifndef L1Trigger_DemonstratorTools_codecs_EtSum_h
#define L1Trigger_DemonstratorTools_codecs_EtSum_h

#include <array>
#include <vector>

#include "ap_int.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "L1Trigger/L1TTrackMatch/interface/L1TkEtMissEmuAlgo.h"

namespace l1t::demo::codecs {

  ap_uint<64> encodeEtSum(const l1t::EtSum& v);

  // Encodes EtSum collection onto 1 'logical' output link
  std::array<std::vector<ap_uint<64>>, 1> encodeEtSums(const edm::View<l1t::EtSum>&);

  std::vector<l1t::EtSum> decodeEtSums(const std::vector<ap_uint<64>>&);

}  // namespace l1t::demo::codecs

#endif
