#ifndef DataFormats_L1TCalorimeterPhase2_DigitizedClusterCorrelator_h
#define DataFormats_L1TCalorimeterPhase2_DigitizedClusterCorrelator_h

#include <ap_int.h>
#include <vector>

namespace l1tp2 {

  class DigitizedClusterCorrelator {
  private:
    // Data
    unsigned long long int clusterData;
    unsigned int idxGCTCard;  // 0, 1, or 2

    // Constants
    static constexpr unsigned int n_towers_eta = 34;  // in GCT card unique region
    static constexpr unsigned int n_towers_phi = 24;  // in GCT card unique region
    static constexpr unsigned int n_crystals_in_tower = 5;
    static constexpr float LSB_PT = 0.5;                 // 0.5 GeV
    static constexpr float ETA_RANGE_ONE_SIDE = 1.4841;  // barrel goes from (-1.4841, +1.4841)
    static constexpr float LSB_ETA = ((2 * ETA_RANGE_ONE_SIDE) / (n_towers_eta * n_crystals_in_tower));  // (2.8 / 170)
    static constexpr float LSB_PHI = ((2 * M_PI) / (3 * n_towers_phi * n_crystals_in_tower));            // (2 pi * 360)
    static constexpr float PHI_RANGE_PER_SLR_DEGREES = 120;

    static constexpr unsigned int n_bits_pt = 12;            // 12 bits allocated for pt
    static constexpr unsigned int n_bits_unused_start = 63;  // unused bits start at bit 63

    // "top" of the correlator card #0 in GCT coordinates is iPhi tower index 24
    static constexpr int correlatorCard0_tower_iphi_offset = 68;
    // same but for correlator cards #1 and 2 (cards wrap around phi = 180 degrees):
    static constexpr int correlatorCard1_tower_iphi_offset = 20;
    static constexpr int correlatorCard2_tower_iphi_offset = 44;

  public:
    DigitizedClusterCorrelator() { clusterData = 0x0; }

    DigitizedClusterCorrelator(ap_uint<64> data) { clusterData = data; }

    // Constructor from digitized inputs
    DigitizedClusterCorrelator(ap_uint<12> pt,
                               ap_uint<7> eta,
                               ap_int<7> phi,
                               ap_uint<6> hoe,
                               ap_uint<6> iso,
                               ap_uint<6> shape,
                               ap_uint<3> wp,
                               ap_uint<5> timing,
                               ap_uint<2> brems,
                               ap_uint<10> spare,
                               int iGCTCard) {
      clusterData = ((ap_uint<64>)pt) | (((ap_uint<64>)eta) << 12) | (((ap_uint<64>)(phi & 0x7F)) << 19) |
                    (((ap_uint<64>)hoe) << 26) | (((ap_uint<64>)iso) << 32) | (((ap_uint<64>)shape) << 38) |
                    (((ap_uint<64>)wp) << 44) | (((ap_uint<64>)timing) << 47) | (((ap_uint<64>)brems) << 52) |
                    (((ap_uint<64>)spare) << 54);
      idxGCTCard = iGCTCard;
    }

    ap_uint<64> data() const { return clusterData; }

    // Other getters
    float ptLSB() const { return LSB_PT; }
    ap_uint<12> pt() const { return (clusterData & 0xFFF); }
    float ptFloat() const { return (pt() * ptLSB()); }

    // crystal eta in the correlator region (LSB: 2.8/170)
    ap_uint<7> eta() const { return ((clusterData >> 12) & 0x7F); }  // (seven 1's) 0b11111111 = 0x7F

    // crystal phi in the correlator region (LSB: 2pi/360)
    ap_int<7> phi() const { return ((clusterData >> 19) & 0x7F); }

    // HoE value and flag: not defined yet in the emulator
    ap_uint<6> hoe() const { return ((clusterData >> 26) & 0x3F); }

    ap_uint<6> iso() const { return ((clusterData >> 32) & 0x3F); }  // raw isolation sum

    ap_uint<6> shape() const { return ((clusterData >> 38) & 0x3F); }  // et2x5/et5x5

    ap_uint<3> wp() const { return ((clusterData >> 44) & 0x7); }  // encoded standaloneWP, looseL1TkMatchWP, photonWP

    // timing: not saved in the current emulator
    ap_uint<5> timing() const { return ((clusterData >> 47) & 0x1F); }

    // brems: not saved in the current emulator
    ap_uint<2> brems() const { return ((clusterData >> 52) & 0x3); }

    ap_uint<10> spare() const { return ((clusterData >> 54) & 0x3FF); }

    // which GCT card (0, 1, or 2)
    unsigned int cardNumber() const { return idxGCTCard; }

    // Get real eta (does not depend on card number). crystal iEta = 0 starts at real eta -1.4841.
    // LSB_ETA/2 is to add half a crystal width to get the center of the crystal in eta
    float realEta() const { return (float)((-1 * ETA_RANGE_ONE_SIDE) + (eta() * LSB_ETA) + (LSB_ETA / 2)); }

    // Get real phi (uses card number).
    float realPhi() const {
      // each card starts at a different real phi
      int offset_tower = 0;
      if (cardNumber() == 0) {
        offset_tower = correlatorCard0_tower_iphi_offset;
      } else if (cardNumber() == 1) {
        offset_tower = correlatorCard1_tower_iphi_offset;
      } else if (cardNumber() == 2) {
        offset_tower = correlatorCard2_tower_iphi_offset;
      }

      int tmpphi = (phi() + PHI_RANGE_PER_SLR_DEGREES / 4);
      bool wrapped = !((spare() & 0x2) == 0);
      if (wrapped) {
        tmpphi += PHI_RANGE_PER_SLR_DEGREES / 2;
      }
      int thisPhi = (tmpphi + (offset_tower * n_crystals_in_tower));
      if (thisPhi > 180)
        thisPhi -= 360;  // range between -180 to 180 degrees

      // LSB_PHI/2 is to add half a crystal width to get the center of the crystal in phi
      return (float)((thisPhi * LSB_PHI) + (LSB_PHI / 2));
    }
  };

  // Collection typedef
  typedef std::vector<l1tp2::DigitizedClusterCorrelator> DigitizedClusterCorrelatorCollection;

}  // namespace l1tp2

#endif
