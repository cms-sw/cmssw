
#ifndef L1Trigger_DemonstratorTools_codecs_tkjets_h
#define L1Trigger_DemonstratorTools_codecs_tkjets_h

#include <array>
#include <vector>

#include "ap_int.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/L1TCorrelator/interface/TkJet.h"
#include "DataFormats/L1TCorrelator/interface/TkJetFwd.h"
#include "DataFormats/L1Trigger/interface/TkJetWord.h"

namespace l1t::demo::codecs {

  ap_uint<64> encodeTkJet(const l1t::TkJetWord& t);

  // Encodes TkJet collection onto 1 'logical' output link
  std::array<std::vector<ap_uint<64>>, 1> encodeTkJets(const edm::View<l1t::TkJetWord>&);

  std::vector<l1t::TkJetWord> decodeTkJets(const std::vector<ap_uint<64>>&);

}  // namespace l1t::demo::codecs

#endif
