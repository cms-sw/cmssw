#include "EventFilter/CSCRawToDigi/interface/CSCTMBHeader2020_CCLUT.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader.h"
#include "FWCore/Utilities/interface/Exception.h"

const std::vector<std::pair<unsigned, unsigned> >
    run3_pattern_lookup_tbl = {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4},  /// Valid LCT0, invalid LCT1 combination. Check LCT1 vpf
                               {0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 0}, {1, 1}, {1, 2}, {1, 3},
                               {1, 4}, {2, 0}, {2, 1}, {2, 2}, {2, 3}, {2, 4}, {3, 0}, {3, 1}, {3, 2},
                               {3, 3}, {3, 4}, {4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 4}};  /// pattern IDs 30,31 are reserved

const unsigned run2_pattern_lookup_tbl[2][16] = {{10, 10, 10, 8, 8, 8, 6, 6, 6, 4, 4, 4, 2, 2, 2, 2},
                                                 {10, 10, 10, 9, 9, 9, 7, 7, 7, 5, 5, 5, 3, 3, 3, 3}};

CSCTMBHeader2020_CCLUT::CSCTMBHeader2020_CCLUT() {
  bzero(data(), sizeInWords() * 2);
  bits.nHeaderFrames = 42;
  bits.e0bline = 0x6E0B;
  bits.b0cline = 0xDB0C;
  bits.firmRevCode = 0x401;
  bits.nTBins = 12;
  bits.nCFEBs = 7;
}

CSCTMBHeader2020_CCLUT::CSCTMBHeader2020_CCLUT(const unsigned short* buf) { memcpy(data(), buf, sizeInWords() * 2); }

void CSCTMBHeader2020_CCLUT::setEventInformation(const CSCDMBHeader& dmbHeader) {
  bits.cscID = dmbHeader.dmbID();
  bits.l1aNumber = dmbHeader.l1a();
  bits.bxnCount = dmbHeader.bxn();
}

///returns CLCT digis
std::vector<CSCCLCTDigi> CSCTMBHeader2020_CCLUT::CLCTDigis(uint32_t idlayer) {
  std::vector<CSCCLCTDigi> result;
  unsigned halfstrip = bits.clct0_key_low + (bits.clct0_key_high << 7);
  unsigned strip = halfstrip % 32;
  // CLCT0 1/4 strip bit
  bool quartstrip = (bits.clct0_xky >> 1) & 0x1;
  // CLCT1 1/8 strip bit
  bool eighthstrip = bits.clct0_xky & 0x1;
  unsigned cfeb = halfstrip / 32;

  /// CLCT0 LR bend and slope are from dedicated header fields
  unsigned run3_pattern = bits.clct0_shape & 0x7;  // 3-bit Run3 CLCT PatternID
  unsigned bend = bits.clct0_LR_bend;
  unsigned slope = bits.clct0_slope;
  unsigned run2_pattern = run2_pattern_lookup_tbl[bend][slope];

  CSCCLCTDigi digi0(bits.clct0_valid,
                    bits.clct0_quality,
                    run2_pattern,
                    1,
                    bend,
                    strip,
                    cfeb,
                    bits.clct_bxn,
                    1,
                    bits.bxnPreTrigger,
                    bits.clct0_comparator_code,
                    CSCCLCTDigi::Version::Run3,
                    quartstrip,
                    eighthstrip,
                    run3_pattern,
                    slope);

  halfstrip = bits.clct1_key_low + (bits.clct1_key_high << 7);
  strip = halfstrip % 32;
  // CLCT0 1/4 strip bit
  quartstrip = (bits.clct1_xky >> 1) & 0x1;
  // CLCT1 1/8 strip bit
  eighthstrip = bits.clct1_xky & 0x1;
  cfeb = halfstrip / 32;

  // CLCT LR bend and slope are from dedicated header fields
  run3_pattern = bits.clct1_shape & 0x7;  // 3-bit Run3 CLCT PatternID
  bend = bits.clct1_LR_bend;
  slope = bits.clct1_slope;
  run2_pattern = run2_pattern_lookup_tbl[bend][slope];

  CSCCLCTDigi digi1(bits.clct1_valid,
                    bits.clct1_quality,
                    run2_pattern,
                    1,
                    bend,
                    strip,
                    cfeb,
                    bits.clct_bxn,
                    2,
                    bits.bxnPreTrigger,
                    bits.clct1_comparator_code,
                    CSCCLCTDigi::Version::Run3,
                    quartstrip,
                    eighthstrip,
                    run3_pattern,
                    slope);

  result.push_back(digi0);
  result.push_back(digi1);
  return result;
}

///returns CorrelatedLCT digis
std::vector<CSCCorrelatedLCTDigi> CSCTMBHeader2020_CCLUT::CorrelatedLCTDigis(uint32_t idlayer) const {
  std::vector<CSCCorrelatedLCTDigi> result;
  /// for the zeroth MPC word:
  unsigned strip = bits.MPC_Muon0_clct_key_halfstrip;  //this goes from 0-223
  unsigned slope = (bits.MPC_Muon0_clct_bend_low & 0x7) | (bits.MPC_Muon0_clct_bend_bit4 << 3);
  unsigned hmt = bits.MPC_Muon_HMT_bit0 | (bits.MPC_Muon_HMT_high << 1);  // HighMultiplicityTrigger
  unsigned clct_pattern_id = bits.MPC_Muon_clct_pattern_low | (bits.MPC_Muon_clct_pattern_bit5 << 4);

  std::pair<unsigned, unsigned> run3_pattern_pair = run3_pattern_lookup_tbl[clct_pattern_id % 30];
  unsigned run2_pattern = run2_pattern_lookup_tbl[bits.MPC_Muon0_clct_LR][slope];
  unsigned run3_pattern = run3_pattern_pair.second & 0x7;

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
                            bits.MPC_Muon0_clct_QuarterStrip,
                            bits.MPC_Muon0_clct_EighthStrip,
                            run3_pattern,
                            slope);
  digi.setHMT(hmt);
  result.push_back(digi);
  /// for the first MPC word:
  strip = bits.MPC_Muon1_clct_key_halfstrip;  //this goes from 0-223
  slope = (bits.MPC_Muon1_clct_bend_low & 0x7) | (bits.MPC_Muon1_clct_bend_bit4 << 3);
  run2_pattern = run2_pattern_lookup_tbl[bits.MPC_Muon1_clct_LR][slope];
  run3_pattern = run3_pattern_pair.first & 0x7;

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
                              bits.MPC_Muon1_clct_QuarterStrip,
                              bits.MPC_Muon1_clct_EighthStrip,
                              run3_pattern,
                              slope);
  digi.setHMT(hmt);
  result.push_back(digi);
  return result;
}

CSCShowerDigi CSCTMBHeader2020_CCLUT::showerDigi(uint32_t idlayer) const {
  unsigned hmt_bits = bits.MPC_Muon_HMT_bit0 | (bits.MPC_Muon_HMT_high << 1);  // HighMultiplicityTrigger bits
  uint16_t cscid = 0;                                                  // ??? What is 4-bits CSC Id in CSshowerDigi
  CSCShowerDigi result(hmt_bits & 0x3, (hmt_bits >> 2) & 0x3, cscid);  // 2-bits intime, 2-bits out of time
  return result;
}

void CSCTMBHeader2020_CCLUT::addALCT0(const CSCALCTDigi& digi) {
  throw cms::Exception("In CSC TMBHeaderFormat 2007, ALCTs belong in  ALCT header");
}

void CSCTMBHeader2020_CCLUT::addALCT1(const CSCALCTDigi& digi) {
  throw cms::Exception("In CSC TMBHeaderFormat 2007, ALCTs belong in  ALCT header");
}

void CSCTMBHeader2020_CCLUT::addCLCT0(const CSCCLCTDigi& digi) {
  unsigned halfStrip = digi.getKeyStrip();
  unsigned pattern = digi.getRun3Pattern();
  bits.clct0_valid = digi.isValid();
  bits.clct0_quality = digi.getQuality();
  bits.clct0_shape = pattern;
  // first 7 bits of halfstrip
  bits.clct0_key_low = halfStrip & (0x7F);
  // most-significant (8th) bit
  bits.clct0_key_high = (halfStrip >> 7) & (0x1);
  bits.clct_bxn = digi.getBX();
  bits.bxnPreTrigger = digi.getFullBX();
  bits.clct0_comparator_code = digi.getCompCode();
  bits.clct0_xky = (digi.getEighthStripBit() & 0x1) + ((digi.getQuartStripBit() & 0x1) << 1);
  bits.clct0_LR_bend = digi.getBend();
  bits.clct0_slope = digi.getSlope();
}

void CSCTMBHeader2020_CCLUT::addCLCT1(const CSCCLCTDigi& digi) {
  unsigned halfStrip = digi.getKeyStrip();
  unsigned pattern = digi.getRun3Pattern();
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
  bits.clct1_comparator_code = digi.getCompCode();
  bits.clct1_xky = (digi.getEighthStripBit() & 0x1) + ((digi.getQuartStripBit() & 0x1) << 1);
  bits.clct1_LR_bend = digi.getBend();
  bits.clct1_slope = digi.getSlope();
}

void CSCTMBHeader2020_CCLUT::addCorrelatedLCT0(const CSCCorrelatedLCTDigi& digi) {
  bits.MPC_Muon0_lct_vpf = digi.isValid();
  bits.MPC_Muon0_alct_key_wire = digi.getKeyWG();
  bits.MPC_Muon0_clct_key_halfstrip = digi.getStrip(2) & 0xFF;
  bits.MPC_Muon0_clct_QuarterStrip = digi.getQuartStripBit() & 0x1;
  bits.MPC_Muon0_clct_EighthStrip = digi.getEighthStripBit() & 0x1;
  bits.MPC_Muon0_lct_quality = digi.getQuality() & 0x7;

  // To restore 5-bits Run3 CLCT Pattern ID first assume and set pattern ID = LCT0 Run3 pattern
  uint16_t run3_pattern = digi.getRun3Pattern();
  bits.MPC_Muon_clct_pattern_low = run3_pattern & 0xF;
  bits.MPC_Muon_clct_pattern_bit5 = (run3_pattern >> 4) & 0x1;
  bits.MPC_Muon0_clct_bend_low = digi.getSlope() & 0x7;
  bits.MPC_Muon0_clct_bend_bit4 = (digi.getSlope() >> 3) & 0x1;
  bits.MPC_Muon0_clct_LR = digi.getBend() & 0x1;
  bits.MPC_Muon_HMT_bit0 = digi.getHMT() & 0x1;
  bits.MPC_Muon_HMT_high = (digi.getHMT() >> 1) & 0x7;
  bits.MPC_Muon_alct_bxn = digi.getBX();
  bits.MPC_Muon0_clct_bx0 = digi.getBX0();
}

void CSCTMBHeader2020_CCLUT::addCorrelatedLCT1(const CSCCorrelatedLCTDigi& digi) {
  bits.MPC_Muon1_lct_vpf = digi.isValid();
  bits.MPC_Muon1_alct_key_wire = digi.getKeyWG();
  bits.MPC_Muon1_clct_key_halfstrip = digi.getStrip(2) & 0xFF;
  bits.MPC_Muon1_clct_QuarterStrip = digi.getQuartStripBit() & 0x1;
  bits.MPC_Muon1_clct_EighthStrip = digi.getEighthStripBit() & 0x1;
  bits.MPC_Muon1_lct_quality = digi.getQuality() & 0x7;

  // To restore 5-bits Run3 CLCT Pattern ID assume that LCT0 pattern ID is already processed
  // and combine LCT1 Run3 pattern to set final 5-bit pattern ID
  if (digi.isValid()) {
    uint16_t clct_pattern_id = bits.MPC_Muon_clct_pattern_low | (bits.MPC_Muon_clct_pattern_bit5 << 4);
    uint16_t run3_pattern = digi.getRun3Pattern();
    clct_pattern_id = (clct_pattern_id + (run3_pattern + 1) * 5) % 30;
    bits.MPC_Muon_clct_pattern_low = clct_pattern_id & 0xF;
    bits.MPC_Muon_clct_pattern_bit5 = (clct_pattern_id >> 4) & 0x1;
  }
  bits.MPC_Muon1_clct_bend_low = digi.getSlope() & 0x7;
  bits.MPC_Muon1_clct_bend_bit4 = (digi.getSlope() >> 3) & 0x1;
  bits.MPC_Muon1_clct_LR = digi.getBend() & 0x1;
  bits.MPC_Muon_HMT_bit0 = digi.getHMT() & 0x1;
  bits.MPC_Muon_HMT_high = (digi.getHMT() >> 1) & 0x7;
  bits.MPC_Muon_alct_bxn = digi.getBX();
  bits.MPC_Muon1_clct_bx0 = digi.getBX0();
}

void CSCTMBHeader2020_CCLUT::addShower(const CSCShowerDigi& digi) {
  uint16_t hmt_bits = (digi.bitsInTime() & 0x3) + ((digi.bitsOutOfTime() & 0x3) << 2);
  bits.MPC_Muon_HMT_bit0 = hmt_bits & 0x1;
  bits.MPC_Muon_HMT_high = (hmt_bits >> 1) & 0x7;
}

void CSCTMBHeader2020_CCLUT::print(std::ostream& os) const {
  os << "...............(O)TMB2020 CCLUT/HMT Header.................."
     << "\n";
  os << std::hex << "BOC LINE " << bits.b0cline << " EOB " << bits.e0bline << "\n";
  os << std::hex << "FW revision: 0x" << bits.firmRevCode << "\n";
  os << std::dec << "fifoMode = " << bits.fifoMode << ", nTBins = " << bits.nTBins << "\n";
  os << "boardID = " << bits.boardID << ", cscID = " << bits.cscID << "\n";
  os << "l1aNumber = " << bits.l1aNumber << ", bxnCount = " << bits.bxnCount << "\n";
  os << "trigSourceVect = " << bits.trigSourceVect << ", run3_trig_df = " << bits.run3_trig_df << ", activeCFEBs = 0x"
     << std::hex << (bits.activeCFEBs | (bits.activeCFEBs_2 << 5)) << ", readCFEBs = 0x" << std::hex
     << (bits.readCFEBs | (bits.readCFEBs_2 << 5)) << std::dec << "\n";
  os << "bxnPreTrigger = " << bits.bxnPreTrigger << "\n";
  os << "tmbMatch = " << bits.tmbMatch << " alctOnly = " << bits.alctOnly << " clctOnly = " << bits.clctOnly << "\n";

  os << "readoutCounter: " << std::dec << bits.readoutCounter << ", buf_q_ovf: " << bits.stackOvf
     << ", sync_err: " << bits.syncError << ", has_buf: " << bits.hasBuf << ", buf_stalled: " << bits.bufFull << "\n";
  os << "r_wr_buf_adr: 0x" << std::hex << bits.r_wr_buf_adr << ", r_wr_buf_ready: " << bits.r_wr_buf_ready
     << ", wr_buf_ready: " << bits.wr_buf_ready << ", buf_q_full: " << bits.buf_q_full
     << ", buf_q_empty: " << bits.buf_q_empty << ",\nr_buf_fence_dist: 0x" << bits.r_buf_fence_dist
     << ", buf_q_ovf_err: " << bits.buf_q_ovf_err << ", buf_q_udf_err: " << bits.buf_q_udf_err
     << ", buf_q_adr_err: " << bits.buf_q_adr_err << ", buf_stalled: " << bits.buf_stalled << ",\nbuf_fence_cnt: 0x"
     << bits.buf_fence_cnt << ", reverse_hs_csc: " << bits.reverse_hs_csc
     << ", reverse_hs_me1a: " << bits.reverse_hs_me1a << ", reverse_hs_me1b: " << bits.reverse_hs_me1b << "\n";
  os << "CLCT Words:\n"
     << " bits.clct0_valid = " << bits.clct0_valid << " bits.clct0_shape = " << bits.clct0_shape
     << " bits.clct0_quality = " << bits.clct0_quality
     << " halfstrip = " << (bits.clct0_key_low + (bits.clct0_key_high << 7)) << "\n";
  os << " bits.clct0_xky = " << bits.clct0_xky << " bits.clct0_comparator_code = " << bits.clct0_comparator_code
     << " bits.clct0_LR_bend = " << bits.clct0_LR_bend << " bits.clct0_slope = " << bits.clct0_slope << "\n";

  os << " bits.clct1_valid = " << bits.clct1_valid << " bits.clct1_shape = " << bits.clct1_shape
     << " bits.clct1_quality = " << bits.clct1_quality
     << " halfstrip = " << (bits.clct1_key_low + (bits.clct1_key_high << 7)) << "\n";
  os << " bits.clct1_xky = " << bits.clct1_xky << " bits.clct1_comparator_code = " << bits.clct1_comparator_code
     << " bits.clct1_LR_bend = " << bits.clct1_LR_bend << " bits.clct1_slope = " << bits.clct1_slope << "\n";

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
     << " HMT = " << (bits.MPC_Muon_HMT_bit0 | (bits.MPC_Muon_HMT_high << 1)) << "\n";
}
