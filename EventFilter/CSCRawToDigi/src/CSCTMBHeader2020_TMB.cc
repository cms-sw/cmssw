#include "EventFilter/CSCRawToDigi/interface/CSCTMBHeader2020_TMB.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader.h"
#include "FWCore/Utilities/interface/Exception.h"

/* /// commented to prevent compilation warning 
   /// use it when copper TMB fw would implement CCLUT features
 
const std::vector<std::pair<unsigned, unsigned> >
    run3_pattern_lookup_tbl = {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4},  /// Valid LCT0, invalid LCT1 combination. Check LCT1 vpf
                               {0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 0}, {1, 1}, {1, 2}, {1, 3},
                               {1, 4}, {2, 0}, {2, 1}, {2, 2}, {2, 3}, {2, 4}, {3, 0}, {3, 1}, {3, 2},
                               {3, 3}, {3, 4}, {4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 4}};  /// pattern IDs 30,31 are reserved

const unsigned run2_pattern_lookup_tbl[2][16] = {{10, 10, 10, 8, 8, 8, 6, 6, 6, 4, 4, 4, 2, 2, 2, 2},
                                                 {10, 10, 10, 9, 9, 9, 7, 7, 7, 5, 5, 5, 3, 3, 3, 3}};
*/

CSCTMBHeader2020_TMB::CSCTMBHeader2020_TMB() {
  bzero(data(), sizeInWords() * 2);
  bits.nHeaderFrames = 42;
  bits.e0bline = 0x6E0B;
  bits.b0cline = 0xDB0C;
  bits.firmRevCode = 0x801;  /// copper TMB hybrid fw Run2 CLCT + Run3 MPC/LCT data format + anode-only HMT (March 2022)
  bits.nTBins = 12;
  bits.nCFEBs = 5;
}

CSCTMBHeader2020_TMB::CSCTMBHeader2020_TMB(const unsigned short* buf) { memcpy(data(), buf, sizeInWords() * 2); }

void CSCTMBHeader2020_TMB::setEventInformation(const CSCDMBHeader& dmbHeader) {
  bits.cscID = dmbHeader.dmbID();
  bits.l1aNumber = dmbHeader.l1a();
  bits.bxnCount = dmbHeader.bxn();
}

///returns CLCT digis
std::vector<CSCCLCTDigi> CSCTMBHeader2020_TMB::CLCTDigis(uint32_t idlayer) {
  std::vector<CSCCLCTDigi> result;
  unsigned halfstrip = bits.clct0_key_low + (bits.clct0_key_high << 7);
  unsigned strip = halfstrip % 32;
  unsigned cfeb = halfstrip / 32;
  unsigned pattern = bits.clct0_shape;
  unsigned bend = pattern & 0x1;

  CSCCLCTDigi digi0(
      bits.clct0_valid, bits.clct0_quality, pattern, 1, bend, strip, cfeb, bits.clct_bxn, 1, bits.bxnPreTrigger);

  halfstrip = bits.clct1_key_low + (bits.clct1_key_high << 7);
  strip = halfstrip % 32;
  cfeb = halfstrip / 32;
  pattern = bits.clct1_shape;
  bend = pattern & 0x1;

  CSCCLCTDigi digi1(
      bits.clct1_valid, bits.clct1_quality, pattern, 1, bend, strip, cfeb, bits.clct_bxn, 2, bits.bxnPreTrigger);
  result.push_back(digi0);
  result.push_back(digi1);
  return result;
}

///returns CorrelatedLCT digis
std::vector<CSCCorrelatedLCTDigi> CSCTMBHeader2020_TMB::CorrelatedLCTDigis(uint32_t idlayer) const {
  std::vector<CSCCorrelatedLCTDigi> result;
  unsigned strip = bits.MPC_Muon0_clct_key_halfstrip;  //this goes from 0-223

  /// For TMB Hybrid fw slope is 0 instead of Run3 (bits.MPC_Muon0_clct_bend_low & 0x7) | (bits.MPC_Muon0_clct_bend_bit4 << 3);
  /// 1/4 and 1/8 strips flags are 0
  unsigned slope = 0;
  unsigned hmt = bits.MPC_Muon_HMT_bit0 | (bits.MPC_Muon_HMT_high << 1);  // HighMultiplicityTrigger
  /* /// Run3 format, when full-featured  Run3 copper TMB firmware will be available
  unsigned clct_pattern_id = bits.MPC_Muon_clct_pattern_low | (bits.MPC_Muon_clct_pattern_bit5 << 4);
  std::pair<unsigned, unsigned> run3_pattern_pair = run3_pattern_lookup_tbl[clct_pattern_id % 30];
  unsigned run2_pattern = run2_pattern_lookup_tbl[bits.MPC_Muon0_clct_LR][slope];
  unsigned run3_pattern = run3_pattern_pair.second & 0x7;
  */
  /// For TMB Hybrid fw run3 pattern is set to 0
  unsigned run3_pattern = 0;
  /// For TMB Hybrid fw run2_pattern is directly set in Run3 4-bits bend/slope
  unsigned run2_pattern = (bits.MPC_Muon0_clct_bend_low & 0x7) | (bits.MPC_Muon0_clct_bend_bit4 << 3);

  CSCCorrelatedLCTDigi digi(1,
                            bits.MPC_Muon0_lct_vpf,
                            bits.MPC_Muon0_lct_quality,
                            bits.MPC_Muon0_alct_key_wire,
                            strip,
                            run2_pattern,
                            bits.MPC_Muon0_clct_LR,
                            bits.MPC_Muon_alct_bxn,
                            0,
                            bits.MPC_Muon0_clct_bx0,
                            0,
                            0,
                            CSCCorrelatedLCTDigi::Version::Run3,
                            false,  /// 1/4 strip flag
                            false,  /// 1/8 strip flag
                            run3_pattern,
                            slope);
  digi.setHMT(hmt);
  result.push_back(digi);
  /// for the first MPC word:
  strip = bits.MPC_Muon1_clct_key_halfstrip;  //this goes from 0-223
  /* /// Run3 format, when full-featured  Run3 copper TMB firmware will be available
  slope = (bits.MPC_Muon1_clct_bend_low & 0x7) | (bits.MPC_Muon1_clct_bend_bit4 << 3);
  run2_pattern = run2_pattern_lookup_tbl[bits.MPC_Muon1_clct_LR][slope];
  run3_pattern = run3_pattern_pair.first & 0x7;
  run2_pattern = (bits.MPC_Muon1_clct_bend_low & 0x7) | (bits.MPC_Muon1_clct_bend_bit4 << 3);
  */
  /// For TMB Hybrid fw run2_pattern is directly set in Run3 4-bits bend/slope
  run2_pattern = (bits.MPC_Muon1_clct_bend_low & 0x7) | (bits.MPC_Muon1_clct_bend_bit4 << 3);
  digi = CSCCorrelatedLCTDigi(2,
                              bits.MPC_Muon1_lct_vpf,
                              bits.MPC_Muon1_lct_quality,
                              bits.MPC_Muon1_alct_key_wire,
                              strip,
                              run2_pattern,
                              bits.MPC_Muon1_clct_LR,
                              bits.MPC_Muon_alct_bxn,
                              0,
                              bits.MPC_Muon1_clct_bx0,
                              0,
                              0,
                              CSCCorrelatedLCTDigi::Version::Run3,
                              false,  /// 1/4 strip flag
                              false,  /// 1/8 strip flag
                              run3_pattern,
                              slope);
  digi.setHMT(hmt);
  result.push_back(digi);
  return result;
}

CSCShowerDigi CSCTMBHeader2020_TMB::showerDigi(uint32_t idlayer) const {
  unsigned hmt_bits = bits.MPC_Muon_HMT_bit0 | (bits.MPC_Muon_HMT_high << 1);  // HighMultiplicityTrigger bits
  uint16_t cscid = bits.cscID;  // ??? What is 4-bits CSC Id in CSshowerDigi
  //L1A_TMB_WINDOW is not included in below formula
  //correct version:  CSCConstants::LCT_CENTRAL_BX - bits.pop_l1a_match_win + L1A_TMB_WINDOW/2;
  // same for anode HMT and cathode HMT
  uint16_t bx = CSCConstants::LCT_CENTRAL_BX - bits.pop_l1a_match_win;
  //LCTshower with showerType = 3. wireNHits is not avaiable
  //TMB LCT shower is copied from ALCT shower
  CSCShowerDigi result(hmt_bits & 0x3,
                       (hmt_bits >> 2) & 0x3,
                       cscid,
                       bx,
                       CSCShowerDigi::ShowerType::kLCTShower,
                       0,
                       0);  // 2-bits intime, 2-bits out of time
  return result;
}

CSCShowerDigi CSCTMBHeader2020_TMB::anodeShowerDigi(uint32_t idlayer) const {
  uint16_t cscid = bits.cscID;
  uint16_t bx = CSCConstants::LCT_CENTRAL_BX - bits.pop_l1a_match_win;
  //ALCTshower with showerType = 1. wireNHits is not avaiable
  CSCShowerDigi result(
      bits.anode_hmt & 0x3, 0, cscid, bx, CSCShowerDigi::ShowerType::kALCTShower, 0, 0);  // 2-bits intime, no out of time
  return result;
}

CSCShowerDigi CSCTMBHeader2020_TMB::cathodeShowerDigi(uint32_t idlayer) const {
  uint16_t cscid = bits.cscID;
  uint16_t bx = CSCConstants::LCT_CENTRAL_BX - bits.pop_l1a_match_win;
  //CLCTshower with showerType = 2. comparatorNhits is not avaiable for TMB yet
  CSCShowerDigi result(bits.cathode_hmt & 0x3,
                       0,
                       cscid,
                       bx,
                       CSCShowerDigi::ShowerType::kCLCTShower,
                       0,
                       0);  // 2-bits intime, no out of time
  return result;
}

void CSCTMBHeader2020_TMB::addALCT0(const CSCALCTDigi& digi) {
  throw cms::Exception("In CSC TMBHeaderFormat 2007, ALCTs belong in  ALCT header");
}

void CSCTMBHeader2020_TMB::addALCT1(const CSCALCTDigi& digi) {
  throw cms::Exception("In CSC TMBHeaderFormat 2007, ALCTs belong in  ALCT header");
}

void CSCTMBHeader2020_TMB::addCLCT0(const CSCCLCTDigi& digi) {
  unsigned halfStrip = digi.getKeyStrip();
  unsigned pattern = digi.getPattern();
  bits.clct0_valid = digi.isValid();
  bits.clct0_quality = digi.getQuality();
  bits.clct0_shape = pattern;
  // first 7 bits of halfstrip
  bits.clct0_key_low = halfStrip & (0x7F);
  // most-significant (8th) bit
  bits.clct0_key_high = (halfStrip >> 7) & (0x1);
  bits.clct_bxn = digi.getBX();
  bits.bxnPreTrigger = digi.getFullBX();
}

void CSCTMBHeader2020_TMB::addCLCT1(const CSCCLCTDigi& digi) {
  unsigned halfStrip = digi.getKeyStrip();
  unsigned pattern = digi.getPattern();
  bits.clct1_valid = digi.isValid();
  bits.clct1_quality = digi.getQuality();
  bits.clct1_shape = pattern;
  // first 7 bits of halfstrip
  bits.clct1_key_low = halfStrip & (0x7F);
  // most-significant (8th) bit
  bits.clct1_key_high = (halfStrip >> 7) & (0x1);
  // There is just one BX field common for CLCT0 and CLCT1 (since both
  // are latched at the same BX); set it in addCLCT0().
  bits.bxnPreTrigger = digi.getFullBX();
}

void CSCTMBHeader2020_TMB::addCorrelatedLCT0(const CSCCorrelatedLCTDigi& digi) {
  bits.MPC_Muon0_lct_vpf = digi.isValid();
  bits.MPC_Muon0_alct_key_wire = digi.getKeyWG();
  bits.MPC_Muon0_clct_key_halfstrip = digi.getStrip(2) & 0xFF;
  /// For TMB hybrid fw 1/4 and 1/8 strip flags are 0
  /*
  bits.MPC_Muon0_clct_QuarterStrip = digi.getQuartStripBit() & 0x1;
  bits.MPC_Muon0_clct_EighthStrip = digi.getEighthStripBit() & 0x1;
  */
  bits.MPC_Muon0_clct_QuarterStrip = 0;
  bits.MPC_Muon0_clct_EighthStrip = 0;
  bits.MPC_Muon0_lct_quality = digi.getQuality() & 0x7;

  /// For TMB hybrid fw 5-bits clct_pattern and run3_pattern are 0
  /*
  // To restore 5-bits Run3 CLCT Pattern ID first assume and set pattern ID = LCT0 Run3 pattern
  uint16_t run3_pattern = digi.getRun3Pattern();
  bits.MPC_Muon_clct_pattern_low = run3_pattern & 0xF;
  bits.MPC_Muon_clct_pattern_bit5 = (run3_pattern >> 4) & 0x1;
  */
  bits.MPC_Muon_clct_pattern_low = 0;
  bits.MPC_Muon_clct_pattern_bit5 = 0;
  /// For TMB hybrid fw use run2 pattern ID to fill run3 4-bits bend/slope field
  bits.MPC_Muon0_clct_bend_low = digi.getPattern() & 0x7;
  bits.MPC_Muon0_clct_bend_bit4 = (digi.getPattern() >> 3) & 0x1;
  bits.MPC_Muon0_clct_LR = digi.getBend() & 0x1;
  bits.MPC_Muon_HMT_bit0 = digi.getHMT() & 0x1;
  bits.MPC_Muon_HMT_high = (digi.getHMT() >> 1) & 0x7;
  bits.MPC_Muon_alct_bxn = digi.getBX();
  bits.MPC_Muon0_clct_bx0 = digi.getBX0();
}

void CSCTMBHeader2020_TMB::addCorrelatedLCT1(const CSCCorrelatedLCTDigi& digi) {
  bits.MPC_Muon1_lct_vpf = digi.isValid();
  bits.MPC_Muon1_alct_key_wire = digi.getKeyWG();
  bits.MPC_Muon1_clct_key_halfstrip = digi.getStrip(2) & 0xFF;
  /// For TMB hybrid fw 1/4 and 1/8 strip flags are 0
  /*
  bits.MPC_Muon1_clct_QuarterStrip = digi.getQuartStripBit() & 0x1;
  bits.MPC_Muon1_clct_EighthStrip = digi.getEighthStripBit() & 0x1;
  */
  bits.MPC_Muon1_clct_QuarterStrip = 0;
  bits.MPC_Muon1_clct_EighthStrip = 0;
  bits.MPC_Muon1_lct_quality = digi.getQuality() & 0x7;

  /// For TMB hybrid fw 5-bits clct_pattern and run3_pattern are 0
  /*
  // To restore 5-bits Run3 CLCT Pattern ID assume that LCT0 pattern ID is already processed
  // and combine LCT1 Run3 pattern to set final 5-bit pattern ID
  if (digi.isValid()) {
    uint16_t clct_pattern_id = bits.MPC_Muon_clct_pattern_low | (bits.MPC_Muon_clct_pattern_bit5 << 4);
    uint16_t run3_pattern = digi.getRun3Pattern();
    clct_pattern_id = (clct_pattern_id + (run3_pattern + 1) * 5) % 30;
    bits.MPC_Muon_clct_pattern_low = clct_pattern_id & 0xF;
    bits.MPC_Muon_clct_pattern_bit5 = (clct_pattern_id >> 4) & 0x1;
  }
  */
  bits.MPC_Muon_clct_pattern_low = 0;
  bits.MPC_Muon_clct_pattern_bit5 = 0;
  /// For TMB hybrid fw use run2 pattern ID to fill run3 4-bits bend/slope field
  bits.MPC_Muon1_clct_bend_low = digi.getPattern() & 0x7;
  bits.MPC_Muon1_clct_bend_bit4 = (digi.getPattern() >> 3) & 0x1;
  bits.MPC_Muon1_clct_LR = digi.getBend() & 0x1;
  bits.MPC_Muon_HMT_bit0 = digi.getHMT() & 0x1;
  bits.MPC_Muon_HMT_high = (digi.getHMT() >> 1) & 0x7;
  bits.MPC_Muon_alct_bxn = digi.getBX();
  bits.MPC_Muon1_clct_bx0 = digi.getBX0();
}

void CSCTMBHeader2020_TMB::addShower(const CSCShowerDigi& digi) {
  uint16_t hmt_bits = (digi.bitsInTime() & 0x3) + ((digi.bitsOutOfTime() & 0x3) << 2);
  //not valid LCT shower, then in-time bits must be 0
  if (not digi.isValid())
    hmt_bits = ((digi.bitsOutOfTime() & 0x3) << 2);
  bits.MPC_Muon_HMT_bit0 = hmt_bits & 0x1;
  bits.MPC_Muon_HMT_high = (hmt_bits >> 1) & 0x7;
  if (digi.isValid())
    bits.pop_l1a_match_win = CSCConstants::LCT_CENTRAL_BX - digi.getBX();
  else
    bits.pop_l1a_match_win = 3;  //default value
}

void CSCTMBHeader2020_TMB::addAnodeShower(const CSCShowerDigi& digi) {
  uint16_t hmt_bits = digi.bitsInTime() & 0x3;
  if (not digi.isValid())
    hmt_bits = 0;
  bits.anode_hmt = hmt_bits;
  if (digi.isValid())
    bits.pop_l1a_match_win = CSCConstants::LCT_CENTRAL_BX - digi.getBX();
  else
    bits.pop_l1a_match_win = 3;  //default value
}

void CSCTMBHeader2020_TMB::addCathodeShower(const CSCShowerDigi& digi) {
  /// For TMB hybrid fw cathode HMT bits are 0
  /* uint16_t hmt_bits = digi.bitsInTime() & 0x3;
  bits.cathode_hmt = hmt_bits;
  */
  bits.cathode_hmt = 0;
}

void CSCTMBHeader2020_TMB::print(std::ostream& os) const {
  os << "...............(O)TMB2020 TMB Run3 Header.................."
     << "\n";
  os << std::hex << "BOC LINE " << bits.b0cline << " EOB " << bits.e0bline << "\n";
  os << std::hex << "FW revision: 0x" << bits.firmRevCode << "\n";
  os << std::dec << "fifoMode = " << bits.fifoMode << ", nTBins = " << bits.nTBins << "\n";
  os << "boardID = " << bits.boardID << ", cscID = " << bits.cscID << "\n";
  os << "l1aNumber = " << bits.l1aNumber << ", bxnCount = " << bits.bxnCount << "\n";
  os << "trigSourceVect = " << bits.trigSourceVect << ", activeCFEBs = 0x" << std::hex
     << (bits.activeCFEBs | (bits.activeCFEBs_2 << 5)) << ", readCFEBs = 0x" << std::hex
     << (bits.readCFEBs | (bits.readCFEBs_2 << 5)) << std::dec << "\n";
  os << "bxnPreTrigger = " << bits.bxnPreTrigger << "\n";
  os << "ALCT location in CLCT window " << bits.matchWin << " L1A location in TMB window " << bits.pop_l1a_match_win
     << "\n";
  os << "tmbMatch = " << bits.tmbMatch << " alctOnly = " << bits.alctOnly << " clctOnly = " << bits.clctOnly << "\n";

  os << "CLCT Words:\n"
     << " bits.clct0_valid = " << bits.clct0_valid << " bits.clct0_shape = " << bits.clct0_shape
     << " bits.clct0_quality = " << bits.clct0_quality
     << " halfstrip = " << (bits.clct0_key_low + (bits.clct0_key_high << 7)) << "\n";

  os << " bits.clct1_valid = " << bits.clct1_valid << " bits.clct1_shape = " << bits.clct1_shape
     << " bits.clct1_quality = " << bits.clct1_quality
     << " halfstrip = " << (bits.clct1_key_low + (bits.clct1_key_high << 7)) << "\n";

  os << "MPC Words:\n"
     << " LCT0 valid = " << bits.MPC_Muon0_lct_vpf << " key WG = " << bits.MPC_Muon0_alct_key_wire
     << " key halfstrip = " << bits.MPC_Muon0_clct_key_halfstrip
     << " 1/4strip flag = " << bits.MPC_Muon0_clct_QuarterStrip
     << " 1/8strip flag = " << bits.MPC_Muon0_clct_EighthStrip << "\n"
     << " quality = " << bits.MPC_Muon0_lct_quality
     << " slope/bend = " << ((bits.MPC_Muon0_clct_bend_low & 0x7) | (bits.MPC_Muon0_clct_bend_bit4 << 3))
     << " L/R bend = " << bits.MPC_Muon0_clct_LR << "\n";

  os << " LCT1 valid = " << bits.MPC_Muon1_lct_vpf << " key WG = " << bits.MPC_Muon1_alct_key_wire
     << " key halfstrip = " << bits.MPC_Muon1_clct_key_halfstrip
     << " 1/4strip flag = " << bits.MPC_Muon1_clct_QuarterStrip
     << " 1/8strip flag = " << bits.MPC_Muon1_clct_EighthStrip << "\n"
     << " quality = " << bits.MPC_Muon1_lct_quality
     << " slope/bend = " << ((bits.MPC_Muon1_clct_bend_low & 0x7) | (bits.MPC_Muon1_clct_bend_bit4 << 3))
     << " L/R bend = " << bits.MPC_Muon1_clct_LR << "\n";

  os << " clct_5bit_pattern_id = " << (bits.MPC_Muon_clct_pattern_low | (bits.MPC_Muon_clct_pattern_bit5 << 4))
     << " HMT = " << (bits.MPC_Muon_HMT_bit0 | (bits.MPC_Muon_HMT_high << 1)) << ", alctHMT = " << bits.anode_hmt
     << ", clctHMT = " << bits.cathode_hmt << "\n";
}
