#include "EventFilter/CSCRawToDigi/interface/CSCTMBHeader2006.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader.h"
#include "FWCore/Utilities/interface/Exception.h"

CSCTMBHeader2006::CSCTMBHeader2006() {
  bzero(data(), sizeInWords() * 2);
  bits.nHeaderFrames = 26;
  bits.e0bline = 0x6E0B;
  bits.b0cline = 0x6B0C;
  bits.nTBins = 7;
  bits.nCFEBs = 5;
}

CSCTMBHeader2006::CSCTMBHeader2006(const unsigned short* buf) { memcpy(&bits, buf, sizeInWords() * 2); }

void CSCTMBHeader2006::setEventInformation(const CSCDMBHeader& dmbHeader) {
  bits.cscID = dmbHeader.dmbID();
  bits.l1aNumber = dmbHeader.l1a();
  // bits.bxnCount = dmbHeader.bxn();
}

///returns CLCT digis
std::vector<CSCCLCTDigi> CSCTMBHeader2006::CLCTDigis(uint32_t idlayer) {
  std::vector<CSCCLCTDigi> result;
  ///fill digis here
  /// for the zeroth bits.clct:
  int shape = 0;
  int type = 0;

  if (bits.firmRevCode < 3769) {  //3769 is may 25 2007 - date of firmware with halfstrip only patterns
    shape = bits.clct0_shape;
    type = bits.clct0_strip_type;
  } else {  //new firmware only halfstrip pattern => stripType==1 and shape is 4 bits
    shape = (bits.clct0_strip_type << 3) + bits.clct0_shape;
    type = 1;
  }
  int strip = bits.clct0_key;
  int cfeb = (bits.clct0_cfeb_low) | (bits.clct0_cfeb_high << 1);
  int bend = bits.clct0_bend;
  //offlineStripNumbering(strip, cfeb, shape, bend);

  CSCCLCTDigi digi0(
      bits.clct0_valid, bits.clct0_quality, shape, type, bend, strip, cfeb, bits.clct0_bxn, 1, bits.bxnPreTrigger);
  //digi0.setFullBX(bits.bxnPreTrigger);
  result.push_back(digi0);

  /// for the first bits.clct:
  if (bits.firmRevCode < 3769) {
    shape = bits.clct1_shape;
    type = bits.clct1_strip_type;
  } else {
    shape = (bits.clct1_strip_type << 3) + bits.clct1_shape;
    type = 1;
  }

  strip = bits.clct1_key;
  cfeb = (bits.clct1_cfeb_low) | (bits.clct1_cfeb_high << 1);
  bend = bits.clct1_bend;
  //offlineStripNumbering(strip, cfeb, shape, bend);
  CSCCLCTDigi digi1(
      bits.clct1_valid, bits.clct1_quality, shape, type, bend, strip, cfeb, bits.clct1_bxn, 2, bits.bxnPreTrigger);
  //digi1.setFullBX(bits.bxnPreTrigger);
  result.push_back(digi1);
  return result;
}

///returns CorrelatedLCT digis
std::vector<CSCCorrelatedLCTDigi> CSCTMBHeader2006::CorrelatedLCTDigis(uint32_t idlayer) const {
  std::vector<CSCCorrelatedLCTDigi> result;
  /// for the zeroth MPC word:
  int strip = bits.MPC_Muon0_halfstrip_clct_pattern;  //this goes from 0-159
  //offlineHalfStripNumbering(strip);
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
  //offlineHalfStripNumbering(strip);
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

void CSCTMBHeader2006::addALCT0(const CSCALCTDigi& digi) {
  throw cms::Exception("In CSC TMBHeaderFormat 2006, ALCTs belong in ALCT header");
}

void CSCTMBHeader2006::addALCT1(const CSCALCTDigi& digi) {
  throw cms::Exception("In CSC TMBHeaderFormat 2006, ALCTs belong in ALCT header");
}

void CSCTMBHeader2006::addCLCT0(const CSCCLCTDigi& digi) {
  int strip = digi.strip();
  int cfeb = digi.cfeb();
  int bend = digi.bend();
  int pattern = digi.pattern();
  //hardwareStripNumbering(strip, cfeb, pattern, bend);
  bits.clct0_valid = digi.isValid();
  bits.clct0_quality = digi.quality();
  bits.clct0_shape = pattern;
  bits.clct0_strip_type = digi.stripType();
  bits.clct0_bend = bend;
  bits.clct0_key = strip;
  bits.clct0_cfeb_low = (cfeb & 0x1);
  bits.clct0_cfeb_high = (cfeb >> 1);
  bits.clct0_bxn = digi.bx();
  bits.bxnPreTrigger = digi.fullBX();
}

void CSCTMBHeader2006::addCLCT1(const CSCCLCTDigi& digi) {
  int strip = digi.strip();
  int cfeb = digi.cfeb();
  int bend = digi.bend();
  int pattern = digi.pattern();
  //hardwareStripNumbering(strip, cfeb, pattern, bend);
  bits.clct1_valid = digi.isValid();
  bits.clct1_quality = digi.quality();
  bits.clct1_shape = pattern;
  bits.clct1_strip_type = digi.stripType();
  bits.clct1_bend = bend;
  bits.clct1_key = strip;
  bits.clct1_cfeb_low = (cfeb & 0x1);
  bits.clct1_cfeb_high = (cfeb >> 1);
  bits.clct1_bxn = digi.bx();
  bits.bxnPreTrigger = digi.fullBX();
}

void CSCTMBHeader2006::addCorrelatedLCT0(const CSCCorrelatedLCTDigi& digi) {
  int halfStrip = digi.strip();
  //hardwareHalfStripNumbering(halfStrip);

  bits.MPC_Muon0_vpf_ = digi.isValid();
  bits.MPC_Muon0_wire_ = digi.keyWireGroup();
  bits.MPC_Muon0_clct_pattern_ = digi.pattern();
  bits.MPC_Muon0_quality_ = digi.quality();
  bits.MPC_Muon0_halfstrip_clct_pattern = halfStrip;
  bits.MPC_Muon0_bend_ = digi.bend();
  bits.MPC_Muon0_SyncErr_ = digi.syncErr();
  bits.MPC_Muon0_bx_ = digi.bx();
  bits.MPC_Muon0_bc0_ = digi.bx0();
  bits.MPC_Muon0_cscid_low = digi.cscID() & 0x7;
  bits.MPC_Muon0_cscid_bit4 = (digi.cscID() >> 3) & 0x1;
}

void CSCTMBHeader2006::addCorrelatedLCT1(const CSCCorrelatedLCTDigi& digi) {
  int halfStrip = digi.strip();
  //hardwareHalfStripNumbering(halfStrip);

  bits.MPC_Muon1_vpf_ = digi.isValid();
  bits.MPC_Muon1_wire_ = digi.keyWireGroup();
  bits.MPC_Muon1_clct_pattern_ = digi.pattern();
  bits.MPC_Muon1_quality_ = digi.quality();
  bits.MPC_Muon1_halfstrip_clct_pattern = halfStrip;
  bits.MPC_Muon1_bend_ = digi.bend();
  bits.MPC_Muon1_SyncErr_ = digi.syncErr();
  bits.MPC_Muon1_bx_ = digi.bx();
  bits.MPC_Muon1_bc0_ = digi.bx0();
  bits.MPC_Muon1_cscid_low = digi.cscID() & 0x7;
  bits.MPC_Muon0_cscid_bit4 = (digi.cscID() >> 3) & 0x1;
}

void CSCTMBHeader2006::print(std::ostream& os) const {
  os << "...............TMB Header.................."
     << "\n";
  os << std::hex << "BOC LINE " << bits.b0cline << " EOB " << bits.e0bline << "\n";
  os << std::dec << "fifoMode = " << bits.fifoMode << ", nTBins = " << bits.nTBins << "\n";
  os << "dumpCFEBs = " << bits.dumpCFEBs << ", nHeaderFrames = " << bits.nHeaderFrames << "\n";
  os << "boardID = " << bits.boardID << ", cscID = " << bits.cscID << "\n";
  os << "l1aNumber = " << bits.l1aNumber << ", bxnCount = " << bits.bxnCount << "\n";
  os << "preTrigTBins = " << bits.preTrigTBins << ", nCFEBs = " << bits.nCFEBs << "\n";
  os << "trigSourceVect = " << bits.trigSourceVect << ", activeCFEBs = " << bits.activeCFEBs << "\n";
  os << "bxnPreTrigger = " << bits.bxnPreTrigger << "\n";
  os << "tmbMatch = " << bits.tmbMatch << " alctOnly = " << bits.alctOnly << " clctOnly = " << bits.clctOnly
     << " alctMatchTime = " << bits.alctMatchTime << "\n";
  os << "hs_thresh = " << bits.hs_thresh << ", ds_thresh = " << bits.ds_thresh << "\n";
  os << ".clct0_key = " << bits.clct0_key << " clct0_shape = " << bits.clct0_shape
     << " clct0_quality = " << bits.clct0_quality << "\n";
  os << "r_buf_nbusy = " << bits.r_buf_nbusy << "\n";
  os << "Firmware Rev code " << bits.firmRevCode << "\n";
  os << "..................CLCT....................." << std::endl;
}

//unsigned short * data()  = 0;
