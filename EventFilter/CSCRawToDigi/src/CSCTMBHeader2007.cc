#include "EventFilter/CSCRawToDigi/interface/CSCTMBHeader2007.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader.h"
#include "FWCore/Utilities/interface/Exception.h"

CSCTMBHeader2007::CSCTMBHeader2007() 
{
    bzero(data(), sizeInWords()*2);
    bits.nHeaderFrames = 42;
    bits.e0bline = 0x6E0B;
    bits.b0cline = 0xDB0C;
    bits.nTBins = 7;
    bits.nCFEBs = 5;
}


CSCTMBHeader2007::CSCTMBHeader2007(const unsigned short * buf)
{
  memcpy(data(), buf, sizeInWords()*2);
}

  
void CSCTMBHeader2007::setEventInformation(const CSCDMBHeader & dmbHeader) 
{
    bits.cscID = dmbHeader.dmbID();
    bits.l1aNumber = dmbHeader.l1a();
//    bits.bxnCount = dmbHeader.bxn();
}

 ///returns CLCT digis
std::vector<CSCCLCTDigi> CSCTMBHeader2007::CLCTDigis(uint32_t idlayer) 
{
    std::vector<CSCCLCTDigi> result;
    int strip   = bits.clct0_key;
    int cfeb    = (bits.clct0_cfeb_low)|(bits.clct0_cfeb_high<<1);
    int pattern = bits.clct0_shape;
    int bend    = bits.clct0_bend;
    //offlineStripNumbering(strip, cfeb, pattern, bend);
    CSCCLCTDigi digi0(bits.clct0_valid, bits.clct0_quality,
                      pattern, 1, bend, strip, cfeb, bits.clct0_bxn, 1, bits.bxnPreTrigger);
    //digi0.setFullBX(bits.bxnPreTrigger);

    strip = bits.clct1_key;
    cfeb = (bits.clct1_cfeb_low)|(bits.clct1_cfeb_high<<1);
    pattern = bits.clct1_shape;
    bend    = bits.clct1_bend;
    //offlineStripNumbering(strip, cfeb, pattern, bend);
    CSCCLCTDigi digi1(bits.clct1_valid, bits.clct1_quality,
                      pattern, 1, bend, strip, cfeb, bits.clct1_bxn, 2, bits.bxnPreTrigger);
    //digi1.setFullBX(bits.bxnPreTrigger);

    //if (digi0.isValid() && digi1.isValid()) swapCLCTs(digi0, digi1);

    result.push_back(digi0);
    result.push_back(digi1);


    return result;
}

 ///returns CorrelatedLCT digis
std::vector<CSCCorrelatedLCTDigi> 
CSCTMBHeader2007::CorrelatedLCTDigis(uint32_t idlayer) const 
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
CSCTMBHeader2007::addALCT0(const CSCALCTDigi & digi)
{
  throw cms::Exception("In CSC TMBHeaderFormat 2007, ALCTs belong in ALCT header");
}


void
CSCTMBHeader2007::addALCT1(const CSCALCTDigi & digi)
{
  throw cms::Exception("In CSC TMBHeaderFormat 2007, ALCTs belong in ALCT header");
}

void
CSCTMBHeader2007::addCLCT0(const CSCCLCTDigi & digi)
{
  int strip = digi.getStrip();
  int cfeb = digi.getCFEB();
  int bend = digi.getBend();
  int pattern = digi.getPattern();
  //hardwareStripNumbering(strip, cfeb, pattern, bend);
  bits.clct0_valid = digi.isValid();
  bits.clct0_quality = digi.getQuality();
  bits.clct0_shape = pattern;
  bits.clct0_bend = bend;
  bits.clct0_key = strip;
  bits.clct0_cfeb_low = (cfeb & 0x1);
  bits.clct0_cfeb_high = (cfeb>>1);
  bits.clct0_bxn = digi.getBX();
  bits.bxnPreTrigger = digi.getFullBX();
  bits.bxnCount = (digi.getFullBX() + 167) & 0xFFF;
}

void
CSCTMBHeader2007::addCLCT1(const CSCCLCTDigi & digi)
{
  int strip = digi.getStrip();
  int cfeb = digi.getCFEB();
  int bend = digi.getBend();
  int pattern = digi.getPattern();
  //hardwareStripNumbering(strip, cfeb, pattern, bend);
  bits.clct1_valid = digi.isValid();
  bits.clct1_quality = digi.getQuality();
  bits.clct1_shape = pattern;
  bits.clct1_bend = bend;
  bits.clct1_key = strip;
  bits.clct1_cfeb_low = (cfeb & 0x1);
  bits.clct1_cfeb_high = (cfeb>>1);
  bits.clct1_bxn = digi.getBX();
  bits.bxnPreTrigger = digi.getFullBX();
}

void
CSCTMBHeader2007::addCorrelatedLCT0(const CSCCorrelatedLCTDigi & digi)
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
CSCTMBHeader2007::addCorrelatedLCT1(const CSCCorrelatedLCTDigi & digi)
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


void CSCTMBHeader2007::print(std::ostream & os) const
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
     << ", activeCFEBs = " << bits.activeCFEBs <<"\n";
  os << "bxnPreTrigger = " << bits.bxnPreTrigger << "\n";
  os << "tmbMatch = " << bits.tmbMatch << " alctOnly = " << bits.alctOnly
     << " clctOnly = " << bits.clctOnly << "\n";
//     << " alctMatchTime = " << alctMatchTime << " ";
//  os << "hs_thresh = " << hs_thresh << ", ds_thresh = " << ds_thresh
//     << " ";
  os << "clct0_key = " << bits.clct0_key << " bits.clct0_shape = " << bits.clct0_shape
     << " clct0_quality = " << bits.clct0_quality << "\n";
//  os << "r_buf_nbusy = " << r_buf_nbusy << " ";

  os << "..................CLCT....................." << "\n";

}

