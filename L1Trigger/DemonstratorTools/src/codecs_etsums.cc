#include "L1Trigger/DemonstratorTools/interface/codecs/etsums.h"
#include "DataFormats/Math/interface/LorentzVector.h"

namespace l1t::demo::codecs {

  ap_uint<64> encodeEtSum(const l1t::EtSum& etSum) {
    l1tmetemu::EtMiss etMiss;
    etMiss.Et.V = etSum.hwPt();
    etMiss.Phi = etSum.hwPhi();
    ap_uint<1> valid = (etSum.hwQual() > 0);
    ap_uint<64 - (l1tmetemu::kMETSize + l1tmetemu::kMETPhiSize + 1)> unassigned = 0;
    ap_uint<64> etSumWord = (unassigned, etMiss.Phi, etMiss.Et.range(), valid);
    return etSumWord;
  }

  // Encodes etsum collection onto 1 output link
  std::array<std::vector<ap_uint<64>>, 1> encodeEtSums(const edm::View<l1t::EtSum>& etSums) {
    std::vector<ap_uint<64>> etSumWords;

    for (const auto& etSum : etSums)
      etSumWords.push_back(encodeEtSum(etSum));

    std::array<std::vector<ap_uint<64>>, 1> linkData;

    for (size_t i = 0; i < linkData.size(); i++) {
      // Pad etsum vectors -> full packet length (48 frames, but only 1 etsum max)
      etSumWords.resize(1, 0);
      linkData.at(i) = etSumWords;
    }

    return linkData;
  }

  std::vector<l1t::EtSum> decodeEtSums(const std::vector<ap_uint<64>>& frames) {
    std::vector<l1t::EtSum> etSums;

    for (const auto& x : frames) {
      if (not x.test(0))
        break;

      math::XYZTLorentzVector v(0, 0, 0, 0);
      l1t::EtSum s(v,
                   l1t::EtSum::EtSumType::kMissingEt,
                   l1tmetemu::METWord_t(x(1 + l1tmetemu::kMETSize, 1)),
                   0,
                   l1tmetemu::METWordphi_t(x(1 + l1tmetemu::kMETSize + l1tmetemu::kMETPhiSize, 17)).to_int(),
                   0);
      etSums.push_back(s);
    }

    return etSums;
  }

}  // namespace l1t::demo::codecs
