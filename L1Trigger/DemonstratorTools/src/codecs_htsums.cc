
#include "L1Trigger/DemonstratorTools/interface/codecs/htsums.h"

namespace l1t::demo::codecs {

  ap_uint<64> encodeHtSum(const l1t::EtSum& etSum) {

  	l1tmhtemu::EtMiss etMiss;
  	etMiss.Et = etSum.p4().energy();
  	etMiss.Phi = etSum.hwPhi();
    l1tmhtemu::Et_t HT = etSum.hwPt();
  	ap_uint<1> valid = (etSum.hwQual() > 0);
  	ap_uint<64 - (l1tmhtemu::kHTSize + l1tmhtemu::kMHTSize + l1tmhtemu::kMHTPhiSize + 1)> unassigned = 0;
  	ap_uint<64> etSumWord = (unassigned, HT, etMiss.Phi, etMiss.Et, valid);
  	return etSumWord;

  }

  // Encodes etsum collection onto 1 output link
  std::array<std::vector<ap_uint<64>>, 1> encodeHtSums(const edm::View<l1t::EtSum>& etSums) {
    std::vector<ap_uint<64>> etSumWords;

    for (const auto& etSum : etSums)
      etSumWords.push_back(encodeHtSum(etSum));

    std::array<std::vector<ap_uint<64>>, 1> linkData;

    for (size_t i = 0; i < linkData.size(); i++) {
      // Pad etsum vectors -> full packet length (48 frames, but only 1 etsum max)
      etSumWords.resize(1, 0);
      linkData.at(i) = etSumWords;
    }

    return linkData;
  }

  std::vector<l1t::EtSum> decodeHtSums(const std::vector<ap_uint<64>>& frames) {
    std::vector<l1t::EtSum> etSums;

    for (const auto& x : frames) {
      if (not x.test(0))
        break;

      math::XYZTLorentzVector v(0, 0, 0, l1tmhtemu::MHT_t(x(l1tmhtemu::kMHTSize + 1, 1)).to_int());
      l1t::EtSum s(v,
                   l1t::EtSum::EtSumType::kMissingHt,
                   l1tmhtemu::Et_t(x(l1tmhtemu::kHTSize + l1tmhtemu::kMHTPhiSize + l1tmhtemu::kMHTSize + 1, 1 + l1tmhtemu::kMHTPhiSize + l1tmhtemu::kMHTSize + 1)).to_int(),
                   0,
                   l1tmhtemu::MHTphi_t(x(l1tmhtemu::kMHTPhiSize + l1tmhtemu::kMHTSize + 1, 1 + l1tmhtemu::kMHTSize + 1)).to_int(), 0);
      etSums.push_back(s);
    }

    return etSums;
  }

}  // namespace l1t::demo::codecs