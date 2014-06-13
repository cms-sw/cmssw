#include "EventFilter/CSCRawToDigi/interface/CSCTMBHeader2013.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader.h"
#include "FWCore/Utilities/interface/Exception.h"

CSCTMBHeader2013::CSCTMBHeader2013() 
{
    bzero(data(), sizeInWords()*2);
    bits.nHeaderFrames = 42;
    bits.e0bline = 0x6E0B;
    bits.b0cline = 0xDB0C;
    bits.firmRevCode = 0x7a76;
    bits.nTBins = 12;
    bits.nCFEBs = 5;
}


CSCTMBHeader2013::CSCTMBHeader2013(const unsigned short * buf)
{
  memcpy(data(), buf, sizeInWords()*2);
}

  
void CSCTMBHeader2013::setEventInformation(const CSCDMBHeader & dmbHeader) 
{
    bits.cscID = dmbHeader.dmbID();
    bits.l1aNumber = dmbHeader.l1a24() & 0xFFF;
//    bits.bxnCount = dmbHeader.bxn12();
}

///returns CLCT digis
std::vector<CSCCLCTDigi> CSCTMBHeader2013::CLCTDigis(uint32_t idlayer) 
{
  std::vector<CSCCLCTDigi> result;
  int halfstrip = bits.clct0_key_low + (bits.clct0_key_high << 7);
  int strip   = halfstrip%32;
  int cfeb    = halfstrip/32;
  int pattern = bits.clct0_shape;
  int bend    = pattern &0x1;

  //offlineStripNumbering(strip, cfeb, pattern, bend);
  CSCCLCTDigi digi0(bits.clct0_valid, bits.clct0_quality,
		    pattern, 1, bend, strip, cfeb, bits.clct_bxn, 1, bits.bxnPreTrigger);
  //digi0.setFullBX(bits.bxnPreTrigger);

  halfstrip = bits.clct1_key_low + (bits.clct1_key_high << 7);
  strip   = halfstrip%32;
  cfeb    = halfstrip/32;
  pattern = bits.clct1_shape;
  bend    = pattern &0x1;

  //offlineStripNumbering(strip, cfeb, pattern, bend);
  CSCCLCTDigi digi1(bits.clct1_valid, bits.clct1_quality,
		    pattern, 1, bend, strip, cfeb, bits.clct_bxn, 2, bits.bxnPreTrigger);
  //digi1.setFullBX(bits.bxnPreTrigger);
  result.push_back(digi0);
  result.push_back(digi1);
  return result;
}

///returns CorrelatedLCT digis
std::vector<CSCCorrelatedLCTDigi> 
CSCTMBHeader2013::CorrelatedLCTDigis(uint32_t idlayer) const 
{
    std::vector<CSCCorrelatedLCTDigi> result;
    /// for the zeroth MPC word:
    int strip = bits.MPC_Muon0_halfstrip_clct_pattern;//this goes from 0-159
    //offlineHalfStripNumbering(strip);
    CSCCorrelatedLCTDigi digi(1, bits.MPC_Muon0_vpf_, bits.MPC_Muon0_quality_,
                              bits.MPC_Muon0_wire_, strip, bits.MPC_Muon0_clct_pattern_,
                              bits.MPC_Muon0_bend_, bits.MPC_Muon0_bx_, 0,
                              bits.MPC_Muon0_bc0_, bits.MPC_Muon0_SyncErr_,
                              bits.MPC_Muon0_cscid_low | (bits.MPC_Muon0_cscid_bit4<<3));
    result.push_back(digi);
    /// for the first MPC word:
    strip = bits.MPC_Muon1_halfstrip_clct_pattern;//this goes from 0-159
    //offlineHalfStripNumbering(strip);
    digi = CSCCorrelatedLCTDigi(2, bits.MPC_Muon1_vpf_, bits.MPC_Muon1_quality_,
                                bits.MPC_Muon1_wire_, strip, bits.MPC_Muon1_clct_pattern_,
                                bits.MPC_Muon1_bend_, bits.MPC_Muon1_bx_, 0,
                                bits.MPC_Muon1_bc0_, bits.MPC_Muon1_SyncErr_,
                                bits.MPC_Muon1_cscid_low | (bits.MPC_Muon1_cscid_bit4<<3));
    result.push_back(digi);
    return result;
}

void
CSCTMBHeader2013::addALCT0(const CSCALCTDigi & digi)
{
  throw cms::Exception("In CSC TMBHeaderFormat 2007, ALCTs belong in  ALCT header");
}


void
CSCTMBHeader2013::addALCT1(const CSCALCTDigi & digi)
{
  throw cms::Exception("In CSC TMBHeaderFormat 2007, ALCTs belong in  ALCT header");
}

void
CSCTMBHeader2013::addCLCT0(const CSCCLCTDigi & digi)
{
  int halfStrip = digi.getKeyStrip();
  int pattern = digi.getPattern();
  //int bend = digi.getBend();
  //hardwareStripNumbering(strip, cfeb, pattern, bend);
  bits.clct0_valid = digi.isValid();
  bits.clct0_quality = digi.getQuality();
  bits.clct0_shape = pattern;
  // first 7 bits of halfstrip
  bits.clct0_key_low = halfStrip & (0x7F);
  // most-significant (8th) bit
  bits.clct0_key_high = (halfStrip >> 7) & (0x1);
  bits.clct_bxn = digi.getBX();
  bits.bxnPreTrigger = digi.getFullBX();
  bits.bxnCount = (digi.getFullBX() + 167) & 0xFFF;
  
}

void
CSCTMBHeader2013::addCLCT1(const CSCCLCTDigi & digi)
{
  int halfStrip = digi.getKeyStrip();
  int pattern = digi.getPattern();
  //int bend = digi.getBend();
  //hardwareStripNumbering(strip, cfeb, pattern, bend);
  bits.clct1_valid = digi.isValid();
  bits.clct1_quality = digi.getQuality();
  bits.clct1_shape = pattern;
  // first 7 bits of halfstrip
  bits.clct1_key_low = halfStrip & (0x7F);
  // most-significant (8th) bit
  bits.clct1_key_high = (halfStrip >> 7) & (0x1);
  // There is just one BX field common for CLCT0 and CLCT1 (since both
  // are latched at the same BX); set it in addCLCT0().
  //bits.clct_bxn = digi.getBX();
  bits.bxnPreTrigger = digi.getFullBX();
}

void
CSCTMBHeader2013::addCorrelatedLCT0(const CSCCorrelatedLCTDigi & digi)
{
  int halfStrip = digi.getStrip();
  //hardwareHalfStripNumbering(halfStrip);

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
  bits.MPC_Muon0_cscid_bit4 = (digi.getCSCID()>>3) & 0x1;
}

void
CSCTMBHeader2013::addCorrelatedLCT1(const CSCCorrelatedLCTDigi & digi)
{
  int halfStrip = digi.getStrip();
  //hardwareHalfStripNumbering(halfStrip);

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
  bits.MPC_Muon1_cscid_bit4 = (digi.getCSCID()>>3) & 0x1;
}


void CSCTMBHeader2013::print(std::ostream & os) const
{
  os << "...............TMB Header.................." << "\n";
  os << std::hex << "BOC LINE " << bits.b0cline << " EOB " << bits.e0bline << "\n";
  os << std::dec << "fifoMode = " << bits.fifoMode
     << ", nTBins = " << bits.nTBins << "\n";
//  os << "dumpCFEBs = " << dumpCFEBs << ", nHeaderFrames = "
//     << nHeaderFrames << "\n";
  os << "boardID = " << bits.boardID << ", cscID = " << bits.cscID << "\n";
  os << "l1aNumber = " << bits.l1aNumber << ", bxnCount = " << bits.bxnCount << "\n";
//  os << "preTrigTBins = " << preTrigTBins << ", nCFEBs = "<< nCFEBs<< " ";
  os << "trigSourceVect = " << bits.trigSourceVect
     << ", activeCFEBs = " << (bits.activeCFEBs | (bits.activeCFEBs_2 << 5))  
     << ", readCFEBs = " << (bits.readCFEBs | (bits.readCFEBs_2 << 5)) << "\n";
  os << "bxnPreTrigger = " << bits.bxnPreTrigger << "\n";
  os << "tmbMatch = " << bits.tmbMatch << " alctOnly = " << bits.alctOnly
     << " clctOnly = " << bits.clctOnly << "\n";
//     << " alctMatchTime = " << alctMatchTime << " ";
//  os << "hs_thresh = " << hs_thresh << ", ds_thresh = " << ds_thresh
//     << " ";
//  os << "clct0_key = " << bits.clct0_key 
  os << " bits.clct0_shape = " << bits.clct0_shape
     << " clct0_quality = " << bits.clct0_quality << "\n";
//  os << "r_buf_nbusy = " << r_buf_nbusy << " ";

  os << "..................CLCT....................." << "\n";

}

