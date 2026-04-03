#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/DemonstratorTools/interface/codecs/tktriplets.h"

namespace l1t::demo::codecs {

  ap_uint<128> encodeTriplet(const l1t::TkTripletWord& t) {
    l1ttripletemu::TkTriplet tkTriplet;
    tkTriplet.valid = t.validWord();
    tkTriplet.pt = t.ptWord();
    tkTriplet.phi = t.phiWord();
    tkTriplet.eta = t.etaWord();
    tkTriplet.mass = t.massWord();
    tkTriplet.trk1Pt = t.trk1PtWord();
    tkTriplet.trk2Pt = t.trk2PtWord();
    tkTriplet.trk3Pt = t.trk3PtWord();
    tkTriplet.charge = t.chargeWord();
    ap_uint<1> valid = (t.validWord());
    ap_uint<128 - (l1ttripletemu::kPtSize + l1ttripletemu::kPhiSize + l1ttripletemu::kEtaSize +
                   l1ttripletemu::kMassSize + 3 * l1ttripletemu::kTrk1PtSize + l1ttripletemu::kChargeSize + 1)>
        unassigned = 0;
    ap_uint<128> tripletWord = (unassigned,
                                tkTriplet.charge.range(),
                                tkTriplet.trk3Pt.range(),
                                tkTriplet.trk2Pt.range(),
                                tkTriplet.trk1Pt.range(),
                                tkTriplet.mass.range(),
                                tkTriplet.eta.range(),
                                tkTriplet.phi.range(),
                                tkTriplet.pt.range(),
                                valid);
    return tripletWord;
  }

  // Output triplet word spans over 2 64-bit link words
  void encodeTripletLinks(std::array<std::vector<ap_uint<128>>, 1>& tripletWords,
                          std::array<std::vector<ap_uint<64>>, 1>& linkData) {
    for (size_t i = 0; i < linkData.size(); i++) {
      tripletWords.at(i).resize(1, 0);
      linkData.at(i).resize(2, {0});

      for (size_t j = 0; j < tripletWords.at(i).size(); j++) {
        linkData.at(i).at(2 * j) = tripletWords.at(i).at(j)(63, 0);
        linkData.at(i).at(2 * j + 1) = tripletWords.at(i).at(j)(127, 64);
      }
    }
  }

  // Encode triplet words in output board format
  std::array<std::vector<ap_uint<64>>, 1> encodeTriplets(const edm::View<l1t::TkTripletWord>& triplets) {
    std::array<std::vector<ap_uint<128>>, 1> tripletWords;

    for (size_t i = 0; i < triplets.size(); i++) {
      tripletWords.at(0).push_back(encodeTriplet(triplets.at(i)));
    }

    std::array<std::vector<ap_uint<64>>, 1> linkData;

    encodeTripletLinks(tripletWords, linkData);

    return linkData;
  }

}  // namespace l1t::demo::codecs
