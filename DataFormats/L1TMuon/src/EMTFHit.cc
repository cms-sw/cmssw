#include "DataFormats/L1TMuon/interface/EMTFHit.h"
#include "DataFormats/CSCDigi/interface/CSCConstants.h"

namespace l1t {

  CSCDetId EMTFHit::CreateCSCDetId() const {
    return CSCDetId((endcap == 1) ? 1 : 2, station, (ring == 4) ? 1 : ring, chamber, 0);
    // Layer always filled as 0 (indicates "whole chamber")
    // See http://cmslxr.fnal.gov/source/L1Trigger/CSCTriggerPrimitives/src/CSCTriggerPrimitivesBuilder.cc#0198
  }

  RPCDetId EMTFHit::CreateRPCDetId() const {
    int roll_ = roll;
    int theta_ = theta_fp / 4;
    if (roll < 1 || roll > 3) {
      switch (station * 2 + ring) {  // Infer roll from integer theta value, by station / ring
        case 4:
          roll_ = (theta_ < 17 ? 3 : (theta_ > 18 ? 1 : 2));
          break;  // RE1/2
        case 6:
          roll_ = (theta_ < 16 ? 3 : (theta_ > 17 ? 1 : 2));
          break;  // RE2/2
        case 8:
          roll_ = (theta_ < 12 ? 3 : (theta_ > 13 ? 1 : 2));
          break;  // RE3/2
        case 9:
          roll_ = (theta_ < 20 ? 3 : (theta_ > 21 ? 1 : 2));
          break;  // RE3/3
        case 10:
          roll_ = (theta_ < 11 ? 3 : (theta_ > 11 ? 1 : 2));
          break;  // RE4/2
        case 11:
          roll_ = (theta_ < 18 ? 3 : (theta_ > 19 ? 1 : 2));
          break;  // RE4/3
        default:
          roll_ = 2;  // Default to 2 if no valid value found
      }
    }

    return RPCDetId(endcap, ring, station, sector_RPC, 1, subsector_RPC, roll_);
    // Layer always filled as 1, as layer 2 is only used in the barrel
  }

  GEMDetId EMTFHit::CreateGEMDetId() const {
    return GEMDetId((endcap == 1) ? 1 : -1, ring, station, layer, chamber, roll);
  }

  ME0DetId EMTFHit::CreateME0DetId() const { return ME0DetId((endcap == 1) ? 1 : -1, layer, chamber, roll); }

  CPPFDigi EMTFHit::CreateCPPFDigi() const {
    if (!Is_RPC())
      return CPPFDigi();

    int board_ = 1 + ((chamber % 36) / 9);  // RPC chamber to CPPF board mapping
    int channel_ = (chamber % 9);           // RPC chamber to CPPF output link mapping
    int sector_ = (sector_idx + 1) - 6 * (endcap == -1);
    int link_ = (neighbor ? 0 : 1 + (((chamber + 33) % 36) % 6));     // RPC chamber to EMTF input link mapping
    int nStrips_ = (strip_low < 0 ? -99 : 1 + strip_hi - strip_low);  // Cluster size in number of strips

    return CPPFDigi(RPC_DetId(),
                    bx,
                    phi_fp / 4,
                    theta_fp / 4,
                    valid,
                    board_,
                    channel_,
                    sector_,
                    link_,
                    strip_low,
                    nStrips_,
                    phi_glob,
                    theta);
  }

  CSCCorrelatedLCTDigi EMTFHit::CreateCSCCorrelatedLCTDigi(const bool isRun3) const {
    CSCCorrelatedLCTDigi lct = CSCCorrelatedLCTDigi(1,
                                                    valid,
                                                    quality,
                                                    wire,
                                                    strip,
                                                    pattern,
                                                    (bend == 1) ? 1 : 0,
                                                    bx + CSCConstants::LCT_CENTRAL_BX,
                                                    0,
                                                    0,
                                                    sync_err,
                                                    csc_ID);
    bool quart_bit = strip_quart_bit == 1;
    bool eighth_bit = strip_eighth_bit == 1;

    lct.setQuartStripBit(quart_bit);
    lct.setEighthStripBit(eighth_bit);
    lct.setSlope(slope);
    lct.setRun3Pattern(pattern_run3);
    lct.setRun3(isRun3);

    return lct;
    // Added Run 3 parameters - EY 04.07.22
    // Filling "trknmb" with 1 and "bx0" with 0 (as in MC).
    // May consider filling "trknmb" with 2 for 2nd LCT in the same chamber. - AWB 24.05.17
    // trknmb and bx0 are unused in the EMTF emulator code. mpclink = 0 (after bx) indicates unsorted.
  }

  // // Not yet implemented - AWB 15.03.17
  // RPCDigi EMTFHit::CreateRPCDigi() const {
  //   return RPCDigi( (strip_hi + strip_lo) / 2, bx + CSCConstants::LCT_CENTRAL_BX );
  // }

  GEMPadDigiCluster EMTFHit::CreateGEMPadDigiCluster() const {
    std::vector<uint16_t> pads;
    for (int i = Pad_low(); i < Pad_hi(); ++i)
      pads.emplace_back(static_cast<uint16_t>(i));
    return GEMPadDigiCluster(pads, bx);
  }

}  // End namespace l1t
