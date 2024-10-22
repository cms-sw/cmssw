
#include "L1Trigger/DemonstratorTools/interface/codecs/tkjets.h"

namespace l1t::demo::codecs {

  ap_uint<64> encodeTkJet(const l1t::TkJetWord& j) { return j.tkJetWord(); }

  // Encodes vertex collection onto 1 output link
  std::array<std::vector<ap_uint<64>>, 1> encodeTkJets(const edm::View<l1t::TkJetWord>& tkJets) {
    std::vector<ap_uint<64>> tkJetWords;

    for (const auto& tkJet : tkJets) {
      tkJetWords.push_back(encodeTkJet(tkJet));
      tkJetWords.push_back(ap_uint<64>(0));
    }

    std::array<std::vector<ap_uint<64>>, 1> linkData;

    for (size_t i = 0; i < linkData.size(); i++) {
      // Pad TkJet vectors -> full packet length (48 frames, but only 12 TkJets max, two words per jet)
      tkJetWords.resize(24, 0);
      linkData.at(i) = tkJetWords;
    }

    return linkData;
  }

  std::vector<l1t::TkJetWord> decodeTkJets(const std::vector<ap_uint<64>>& frames) {
    std::vector<l1t::TkJetWord> tkJets;

    for (size_t f = 0; f < frames.size(); f += 2) {
      // There is no valid bit in the definition right now.
      // Uncomment the next two lines when this is available.
      //if (not x.test(TkJetWord::kValidLSB))
      //  break;

      TkJetWord j(
          TkJetWord::pt_t(frames[f](TkJetWord::TkJetBitLocations::kPtMSB, TkJetWord::TkJetBitLocations::kPtLSB)),
          TkJetWord::glbphi_t(
              frames[f](TkJetWord::TkJetBitLocations::kGlbPhiMSB, TkJetWord::TkJetBitLocations::kGlbPhiLSB)),
          TkJetWord::glbeta_t(
              frames[f](TkJetWord::TkJetBitLocations::kGlbEtaMSB, TkJetWord::TkJetBitLocations::kGlbEtaLSB)),
          TkJetWord::z0_t(frames[f](TkJetWord::TkJetBitLocations::kZ0MSB, TkJetWord::TkJetBitLocations::kZ0LSB)),
          TkJetWord::nt_t(frames[f](TkJetWord::TkJetBitLocations::kNtMSB, TkJetWord::TkJetBitLocations::kNtLSB)),
          TkJetWord::nx_t(frames[f](TkJetWord::TkJetBitLocations::kXtMSB, TkJetWord::TkJetBitLocations::kXtLSB)),
          TkJetWord::dispflag_t(
              frames[f](TkJetWord::TkJetBitLocations::kDispFlagMSB, TkJetWord::TkJetBitLocations::kDispFlagLSB)),
          TkJetWord::tkjetunassigned_t(
              frames[f](TkJetWord::TkJetBitLocations::kUnassignedMSB, TkJetWord::TkJetBitLocations::kUnassignedLSB)));
      tkJets.push_back(j);
    }

    return tkJets;
  }

}  // namespace l1t::demo::codecs
