#include "EventFilter/CSCRawToDigi/interface/CSCTMBHeader2020_Run2.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader.h"
#include "FWCore/Utilities/interface/Exception.h"

CSCTMBHeader2020_Run2::CSCTMBHeader2020_Run2(int firmware_revision) {
  bzero(data(), sizeInWords() * 2);
  bits.nHeaderFrames = 42;
  bits.e0bline = 0x6E0B;
  bits.b0cline = 0xDB0C;
  bits.firmRevCode = firmware_revision;
  bits.nTBins = 12;
  bits.nCFEBs = 5;
}

CSCTMBHeader2020_Run2::CSCTMBHeader2020_Run2(const unsigned short* buf) { memcpy(data(), buf, sizeInWords() * 2); }

void CSCTMBHeader2020_Run2::setEventInformation(const CSCDMBHeader& dmbHeader) {
  bits.cscID = dmbHeader.dmbID();
  bits.l1aNumber = dmbHeader.l1a();
  bits.bxnCount = dmbHeader.bxn();
}

///returns CLCT digis
std::vector<CSCCLCTDigi> CSCTMBHeader2020_Run2::CLCTDigis(uint32_t idlayer) {
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
std::vector<CSCCorrelatedLCTDigi> CSCTMBHeader2020_Run2::CorrelatedLCTDigis(uint32_t idlayer) const {
  std::vector<CSCCorrelatedLCTDigi> result;
  /// for the zeroth MPC word:
  unsigned strip = bits.MPC_Muon0_halfstrip_clct_pattern;  //this goes from 0-159
  CSCCorrelatedLCTDigi digi(1,
                            bits.MPC_Muon0_vpf_,
                            bits.MPC_Muon0_quality_,
                            bits.MPC_Muon0_wire_,
                            strip,
                            bits.MPC_Muon0_clct_pattern_,
                            bits.MPC_Muon0_bend_,
                            bits.MPC_Muon0_bx_,
                            0,
                            bits.MPC_Muon0_bc0_,
                            bits.MPC_Muon0_SyncErr_,
                            bits.MPC_Muon0_cscid_low | (bits.MPC_Muon0_cscid_bit4 << 3));
  result.push_back(digi);
  /// for the first MPC word:
  strip = bits.MPC_Muon1_halfstrip_clct_pattern;  //this goes from 0-159
  digi = CSCCorrelatedLCTDigi(2,
                              bits.MPC_Muon1_vpf_,
                              bits.MPC_Muon1_quality_,
                              bits.MPC_Muon1_wire_,
                              strip,
                              bits.MPC_Muon1_clct_pattern_,
                              bits.MPC_Muon1_bend_,
                              bits.MPC_Muon1_bx_,
                              0,
                              bits.MPC_Muon1_bc0_,
                              bits.MPC_Muon1_SyncErr_,
                              bits.MPC_Muon1_cscid_low | (bits.MPC_Muon1_cscid_bit4 << 3));
  result.push_back(digi);
  return result;
}

void CSCTMBHeader2020_Run2::addALCT0(const CSCALCTDigi& digi) {
  throw cms::Exception("In CSC TMBHeaderFormat 2007, ALCTs belong in  ALCT header");
}

void CSCTMBHeader2020_Run2::addALCT1(const CSCALCTDigi& digi) {
  throw cms::Exception("In CSC TMBHeaderFormat 2007, ALCTs belong in  ALCT header");
}

void CSCTMBHeader2020_Run2::addCLCT0(const CSCCLCTDigi& digi) {
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

void CSCTMBHeader2020_Run2::addCLCT1(const CSCCLCTDigi& digi) {
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

void CSCTMBHeader2020_Run2::addCorrelatedLCT0(const CSCCorrelatedLCTDigi& digi) {
  unsigned halfStrip = digi.getStrip();
  bits.MPC_Muon0_vpf_ = digi.isValid();
  bits.MPC_Muon0_wire_ = digi.getKeyWG();
  bits.MPC_Muon0_clct_pattern_ = digi.getPattern();
  bits.MPC_Muon0_quality_ = digi.getQuality();
  bits.MPC_Muon0_halfstrip_clct_pattern = halfStrip;
  bits.MPC_Muon0_bend_ = digi.getBend();
  bits.MPC_Muon0_SyncErr_ = digi.getSyncErr();
  bits.MPC_Muon0_bx_ = digi.getBX();
  bits.MPC_Muon0_bc0_ = digi.getBX0();
  bits.MPC_Muon0_cscid_low = digi.getCSCID() & 0x7;
  bits.MPC_Muon0_cscid_bit4 = (digi.getCSCID() >> 3) & 0x1;
}

void CSCTMBHeader2020_Run2::addCorrelatedLCT1(const CSCCorrelatedLCTDigi& digi) {
  unsigned halfStrip = digi.getStrip();
  bits.MPC_Muon1_vpf_ = digi.isValid();
  bits.MPC_Muon1_wire_ = digi.getKeyWG();
  bits.MPC_Muon1_clct_pattern_ = digi.getPattern();
  bits.MPC_Muon1_quality_ = digi.getQuality();
  bits.MPC_Muon1_halfstrip_clct_pattern = halfStrip;
  bits.MPC_Muon1_bend_ = digi.getBend();
  bits.MPC_Muon1_SyncErr_ = digi.getSyncErr();
  bits.MPC_Muon1_bx_ = digi.getBX();
  bits.MPC_Muon1_bc0_ = digi.getBX0();
  bits.MPC_Muon1_cscid_low = digi.getCSCID() & 0x7;
  bits.MPC_Muon1_cscid_bit4 = (digi.getCSCID() >> 3) & 0x1;
}

void CSCTMBHeader2020_Run2::print(std::ostream& os) const {
  os << "...............(O)TMB2020 legacy Run2 Header.................."
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
  os << "tmbMatch = " << bits.tmbMatch << " alctOnly = " << bits.alctOnly << " clctOnly = " << bits.clctOnly << "\n";

  os << "CLCT Words:\n"
     << " bits.clct0_valid = " << bits.clct0_valid << " bits.clct0_shape = " << bits.clct0_shape
     << " bits.clct0_quality = " << bits.clct0_quality
     << " halfstrip = " << (bits.clct0_key_low + (bits.clct0_key_high << 7)) << "\n";

  os << " bits.clct1_valid = " << bits.clct1_valid << " bits.clct1_shape = " << bits.clct1_shape
     << " bits.clct1_quality = " << bits.clct1_quality
     << " halfstrip = " << (bits.clct1_key_low + (bits.clct1_key_high << 7)) << "\n";

  os << "MPC Words:\n"
     << " LCT0 valid = " << bits.MPC_Muon0_vpf_ << " key WG = " << bits.MPC_Muon0_wire_
     << " key halfstrip = " << bits.MPC_Muon0_halfstrip_clct_pattern << " pattern = " << bits.MPC_Muon0_clct_pattern_
     << " quality = " << bits.MPC_Muon0_quality_ << "\n";

  os << " LCT1 valid = " << bits.MPC_Muon1_vpf_ << " key WG = " << bits.MPC_Muon1_wire_
     << " key halfstrip = " << bits.MPC_Muon1_halfstrip_clct_pattern << " pattern = " << bits.MPC_Muon1_clct_pattern_
     << " quality = " << bits.MPC_Muon1_quality_ << "\n";
}
