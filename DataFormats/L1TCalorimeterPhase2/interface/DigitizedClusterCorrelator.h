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

    static constexpr unsigned int n_bits_pt = 12;            // 12 bits allocated for pt
    static constexpr unsigned int n_bits_unused_start = 63;  // unused bits start at bit 63

    // "top" of the correlator card #0 in GCT coordinates is iPhi tower index 24
    // Pallabi: changed offset values to correctly calculate realPhi() following L1Trigger/L1CaloTrigger/interface/Phase2L1CaloEGammaUtils.h
    static constexpr int correlatorCard0_tower_iphi_offset = 20;
    // same but for correlator cards #1 and 2 (cards wrap around phi = 180 degrees):
    static constexpr int correlatorCard1_tower_iphi_offset = 44;
    static constexpr int correlatorCard2_tower_iphi_offset = 68;

    // Private member functions to perform digitization
    ap_uint<12> digitizePt(float pt_f) {
      float maxPt_f = (std::pow(2, n_bits_pt) - 1) * LSB_PT;
      // If pT exceeds the maximum (extremely unlikely), saturate the value
      if (pt_f >= maxPt_f) {
        return (ap_uint<12>)0xFFF;
      }

      return (ap_uint<12>)(pt_f / LSB_PT);
    }

    ap_uint<7> digitizeEta(unsigned int iEtaCr) { return (ap_uint<7>)iEtaCr; }

    ap_uint<7> digitizePhi(unsigned int iPhiCr) { return (ap_uint<7>)iPhiCr; }

    // To-do: HoE is not defined for clusters
    ap_uint<6> digitizeHoE(unsigned int hoe) { return (ap_uint<6>)hoe; }

    ap_uint<6> digitizeIso(unsigned int iso) { return (ap_uint<6>)iso; }
    ap_uint<6> digitizeShape(unsigned int shape) { return (ap_uint<6>)shape; }

    // To-do: WP: no information yet
    ap_uint<3> digitizeWP(unsigned int wp) { return (ap_uint<3>)wp; }

    // To-do: timing: no information yet
    ap_uint<5> digitizeTiming(unsigned int timing) { return (ap_uint<5>)timing; }

    // TO-DO: Brems: was brems applied (NOT STORED YET IN GCT)
    ap_uint<2> digitizeBrems(unsigned int brems) { return (ap_uint<2>)brems; }

    // TO-DO: Spare
    ap_uint<10> digitizeSpare(unsigned int spare) { return (ap_uint<10>)spare; }

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
                               int iGCTCard,
                               bool fullydigitizedInputs) {
      (void)fullydigitizedInputs;  // what is this?
      clusterData = ((ap_uint<64>)pt) | (((ap_uint<64>)eta) << 12) | (((ap_uint<64>)(phi & 0x7F)) << 19) |
                    (((ap_uint<64>)hoe) << 26) | (((ap_uint<64>)iso) << 32) |
                    (((ap_uint<64>)shape) << 38) | (((ap_uint<64>)wp) << 44) | (((ap_uint<64>)timing) << 47) |
                    (((ap_uint<64>)brems) << 52) | (((ap_uint<64>)spare) << 54);
      idxGCTCard = iGCTCard;
    }

    ap_uint<64> data() const { return clusterData; }

    // Other getters
    float ptLSB() const { return LSB_PT; }
    ap_uint<12> pt() const { return (clusterData & 0xFFF); }
    float ptFloat() const { return (pt() * ptLSB()); }

    // crystal eta in the correlator region (LSB: 2.8/170)
    ap_uint<7> eta() const { return ((clusterData >> 12) & 0x7F); }  // (eight 1's) 0b11111111 = 0xFF   // not eight but seven?

    // crystal phi in the correlator region (LSB: 2pi/360)
    ap_int<7> phi() const { return ((clusterData >> 19) & 0x7F); }  // (seven 1's) 0b1111111 = 0x7F

    // HoE value and flag: not defined yet in the emulator 
    ap_uint<6> hoe() const { return ((clusterData >> 26) & 0x3F); }      // (four 1's) 0b1111 = 0xF // split 6 bits in 4 and 2?

    // Raw isolation sum: not saved in the emulator
    ap_uint<6> iso() const { return ((clusterData >> 32) & 0x3F); } // split 6 bits in 4 and 2? passes_iso and passes_looseTkiso defined in emulator?

    ap_uint<6> shape() const { return ((clusterData >> 38) & 0x3F); } // shape flags not defined?

    // wp: not saved in the current emulator
    ap_uint<3> wp() const { return ((clusterData >> 44) & 0x7); } //passes_iso and passes_looseTkiso defined in emulator?

    // timing: not saved in the current emulator
    ap_uint<5> timing() const { return ((clusterData >> 47) & 0x1F); }

    // brems: not saved in the current emulator
    ap_uint<2> brems() const { return ((clusterData >> 52) & 0x3); }

    ap_uint<10> spare() const { return ((clusterData >> 54) & 0x3FF); }

    // which GCT card (0, 1, or 2)
    // Pallabi: this is currently set incorrectly in L1Trigger/L1CaloTrigger/plugins/Phase2L1CaloEGammaEmulator.cc so only use it to get realPhi()
    //unsigned int cardNumber() const { return idxGCTCard; }

    // Get real eta (does not depend on card number). crystal iEta = 0 starts at real eta -1.4841.
    // LSB_ETA/2 is to add half a crystal width to get the center of the crystal in eta
    float realEta() const { return (float)((-1 * ETA_RANGE_ONE_SIDE) + (eta() * LSB_ETA) + (LSB_ETA / 2)); }

    // Get real phi (uses card number).
    float realPhi() const {
      // each card starts at a different real phi
      int offset_tower = 0;
      if (idxGCTCard == 0) {
        offset_tower = correlatorCard0_tower_iphi_offset;
      } else if (idxGCTCard == 1) {
        offset_tower = correlatorCard1_tower_iphi_offset;
      } else if (idxGCTCard == 2) {
        offset_tower = correlatorCard2_tower_iphi_offset;
      }

      int thisPhi = (phi() + 30); // add back the offset from L1Trigger/L1CaloTrigger/interface/Phase2L1CaloEGammaUtils.h 
      int tmpphi = thisPhi;
      bool wrapped = ((spare() & 0x2) == 0);
      if (wrapped) { tmpphi = thisPhi + 60; } // add back the offset from L1Trigger/L1CaloTrigger/interface/Phase2L1CaloEGammaUtils.h
      int crPhi = phi() % 5;
      if (phi() < 0) crPhi = (30 + phi()) % 5;
      int towPhi = (tmpphi - crPhi) / 5  + 4; // corrTowPhiOffset = 4

      int iPhi_in_gctCard = (towPhi * 5) + crPhi;
      int globalClusteriPhi = (offset_tower * 5 + iPhi_in_gctCard) % (5 * 72); // CRYSTALS_IN_TOWER_PHI * n_towers_phi
      float size_cell = 2 * M_PI / (5 * 72); // CRYSTALS_IN_TOWER_PHI * n_towers_phi
      return globalClusteriPhi * size_cell - M_PI + 0.00873; // half_crystal_size = 0.00873
    }

    // which GCT card (0, 1, or 2)
    unsigned int cardNumber() const {
      float phiInDegrees = realPhi()* 180 / M_PI + 180;
      int cardnumber = 0;
      if (phiInDegrees > 160 && phiInDegrees < 280) cardnumber = 0;
      if ((phiInDegrees > 280 && phiInDegrees < 360) || phiInDegrees < 40) cardnumber = 1;
      if (phiInDegrees > 40 && phiInDegrees < 160) cardnumber = 2;
      if (pt() == 96) std::cout<<realPhi()<<"\t"<<phiInDegrees<<"\t"<<cardnumber<<std::endl;
      return cardnumber;
    }

    unsigned int slrNumber() const {
      float phiInDegrees = realPhi()* 180 / M_PI + 180;
      int slrnumber = 3;
      if (cardNumber() == 0 && phiInDegrees > 220) slrnumber = 1;
      if (cardNumber() == 1 && (phiInDegrees > 340 || phiInDegrees < 40)) slrnumber = 1;
      if (cardNumber() == 2 && phiInDegrees > 100) slrnumber = 1;
      return slrnumber;
     }

  };

  // Collection typedef
  typedef std::vector<l1tp2::DigitizedClusterCorrelator> DigitizedClusterCorrelatorCollection;

}  // namespace l1tp2

#endif
