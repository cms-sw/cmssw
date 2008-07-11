#include "EventFilter/CSCRawToDigi/interface/CSCTMBHeader2006.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader.h"
#include "FWCore/Utilities/interface/Exception.h"

CSCTMBHeader2006::CSCTMBHeader2006() 
{
    nHeaderFrames = 26;
    bzero(this, sizeInWords()*2);
    e0bline = 0x6E0B;
    b0cline = 0x6B0C;
    nTBins = 7;
    nCFEBs = 5;
}


CSCTMBHeader2006::CSCTMBHeader2006(const unsigned short * buf)
{
  memcpy(data(), buf, sizeInWords()*2);
  print(std::cout);
}

void CSCTMBHeader2006::setEventInformation(const CSCDMBHeader & dmbHeader) 
{
    cscID = dmbHeader.dmbID();
    l1aNumber = dmbHeader.l1a();
    bxnCount = dmbHeader.bxn();
}

 ///returns CLCT digis
std::vector<CSCCLCTDigi> CSCTMBHeader2006::CLCTDigis(uint32_t idlayer) 
{
    std::vector<CSCCLCTDigi> result;
    ///fill digis here
    /// for the zeroth clct:
    int shape=0;
    int type=0;

    if ( firmRevCode < 3769 ) { //3769 is may 25 2007 - date of firmware with halfstrip only patterns
    shape = clct0_shape;
    type  = clct0_strip_type;
    }else {//new firmware only halfstrip pattern => stripType==1 and shape is 4 bits
      shape = ( clct0_strip_type<<3)+clct0_shape;
      type = 1;
    }
    int strip = clct0_key;
    int cfeb = (clct0_cfeb_low)|(clct0_cfeb_high<<1);
    int bend = clct0_bend;
    //offlineStripNumbering(strip, cfeb, shape, bend);

    CSCCLCTDigi digi0(clct0_valid, clct0_quality, shape,
                      type, bend, strip, cfeb, clct0_bxn, 1);
    digi0.setFullBX(bxnPreTrigger);
    result.push_back(digi0);

    /// for the first clct:
    if ( firmRevCode < 3769 ) {
      shape = clct1_shape;
      type  = clct1_strip_type;
    } else {
      shape = (clct1_strip_type<<3)+clct1_shape;
      type = 1;
    }

    strip = clct1_key;
    cfeb = (clct1_cfeb_low)|(clct1_cfeb_high<<1);
    bend = clct1_bend;
    //offlineStripNumbering(strip, cfeb, shape, bend);
    CSCCLCTDigi digi1(clct1_valid, clct1_quality, shape,
                      type, bend, strip, cfeb, clct1_bxn, 2);
    digi1.setFullBX(bxnPreTrigger);
    result.push_back(digi1);
    return result;
}

 ///returns CorrelatedLCT digis
std::vector<CSCCorrelatedLCTDigi> CSCTMBHeader2006::CorrelatedLCTDigis(uint32_t idlayer) const 
{
    std::vector<CSCCorrelatedLCTDigi> result;
    /// for the zeroth MPC word:
    int strip = MPC_Muon0_halfstrip_clct_pattern;//this goes from 0-159
    //offlineHalfStripNumbering(strip);
    CSCCorrelatedLCTDigi digi(1, MPC_Muon0_vpf_, MPC_Muon0_quality_,
                              MPC_Muon0_wire_, strip, MPC_Muon0_clct_pattern_,
                              MPC_Muon0_bend_, MPC_Muon0_bx_, 0,
                              MPC_Muon0_bc0_, MPC_Muon0_SyncErr_,
                              MPC_Muon0_cscid_low | (MPC_Muon0_cscid_bit4<<3) );
    result.push_back(digi);
    /// for the first MPC word:
    strip = MPC_Muon1_halfstrip_clct_pattern;//this goes from 0-159
    //offlineHalfStripNumbering(strip);
    digi = CSCCorrelatedLCTDigi(2, MPC_Muon1_vpf_, MPC_Muon1_quality_,
                                MPC_Muon1_wire_, strip, MPC_Muon1_clct_pattern_,
                                MPC_Muon1_bend_, MPC_Muon1_bx_, 0,
                                MPC_Muon1_bc0_, MPC_Muon1_SyncErr_,
                                MPC_Muon1_cscid_low | (MPC_Muon1_cscid_bit4<<3) );
    result.push_back(digi);
    return result;
}

void
CSCTMBHeader2006::addALCT0(const CSCALCTDigi & digi)
{
  throw cms::Exception("In CSC TMBHeaderFormat 2006, ALCTs belong in ALCT header");
}


void
CSCTMBHeader2006::addALCT1(const CSCALCTDigi & digi)
{
  throw cms::Exception("In CSC TMBHeaderFormat 2006, ALCTs belong in ALCT header");
}

void
CSCTMBHeader2006::addCLCT0(const CSCCLCTDigi & digi)
{
  int strip = digi.getStrip();
  int cfeb = digi.getCFEB();
  int bend = digi.getBend();
  int pattern = digi.getPattern();
  //hardwareStripNumbering(strip, cfeb, pattern, bend);
  clct0_valid = digi.isValid();
  clct0_quality = digi.getQuality();
  clct0_shape = pattern;
  clct0_strip_type = digi.getStripType();
  clct0_bend = bend;
  clct0_key = strip;
  clct0_cfeb_low = (cfeb & 0x1);
  clct0_cfeb_high = (cfeb>>1);
  clct0_bxn = digi.getBX();
  bxnPreTrigger = digi.getFullBX();
}

void
CSCTMBHeader2006::addCLCT1(const CSCCLCTDigi & digi)
{
  int strip = digi.getStrip();
  int cfeb = digi.getCFEB();
  int bend = digi.getBend();
  int pattern = digi.getPattern();
  //hardwareStripNumbering(strip, cfeb, pattern, bend);
  clct1_valid = digi.isValid();
  clct1_quality = digi.getQuality();
  clct1_shape = pattern;
  clct1_strip_type = digi.getStripType();
  clct1_bend = bend;
  clct1_key = strip;
  clct1_cfeb_low = (cfeb & 0x1);
  clct1_cfeb_high = (cfeb>>1);
  clct1_bxn = digi.getBX();
  bxnPreTrigger = digi.getFullBX();
}

void
CSCTMBHeader2006::addCorrelatedLCT0(const CSCCorrelatedLCTDigi & digi)
{
  int halfStrip = digi.getStrip();
  //hardwareHalfStripNumbering(halfStrip);

  MPC_Muon0_vpf_ = digi.isValid();
  MPC_Muon0_wire_ = digi.getKeyWG();
  MPC_Muon0_clct_pattern_ = digi.getPattern();
  MPC_Muon0_quality_ = digi.getQuality();
  MPC_Muon0_halfstrip_clct_pattern = halfStrip;
  MPC_Muon0_bend_ = digi.getBend();
  MPC_Muon0_SyncErr_ = digi.getSyncErr();
  MPC_Muon0_bx_ = digi.getBX();
  MPC_Muon0_bc0_ = digi.getBX0();
  MPC_Muon0_cscid_low = digi.getCSCID();
}

void
CSCTMBHeader2006::addCorrelatedLCT1(const CSCCorrelatedLCTDigi & digi)
{
  int halfStrip = digi.getStrip();
  //hardwareHalfStripNumbering(halfStrip);

  MPC_Muon1_vpf_ = digi.isValid();
  MPC_Muon1_wire_ = digi.getKeyWG();
  MPC_Muon1_clct_pattern_ = digi.getPattern();
  MPC_Muon1_quality_ = digi.getQuality();
  MPC_Muon1_halfstrip_clct_pattern = halfStrip;
  MPC_Muon1_bend_ = digi.getBend();
  MPC_Muon1_SyncErr_ = digi.getSyncErr();
  MPC_Muon1_bx_ = digi.getBX();
  MPC_Muon1_bc0_ = digi.getBX0();
  MPC_Muon1_cscid_low = digi.getCSCID();
}


void CSCTMBHeader2006::print(std::ostream & os) const
{
  os << "...............TMB Header.................." << "\n";
  os << std::hex << "BOC LINE " << b0cline << " EOB " << e0bline << "\n";
  os << std::dec << "fifoMode = " << fifoMode
     << ", nTBins = " << nTBins << "\n";
  os << "dumpCFEBs = " << dumpCFEBs << ", nHeaderFrames = "
     << nHeaderFrames << "\n";
  os << "boardID = " << boardID << ", cscID = " << cscID << "\n";
  os << "l1aNumber = " << l1aNumber << ", bxnCount = " << bxnCount << "\n";
  os << "preTrigTBins = " << preTrigTBins << ", nCFEBs = "<< nCFEBs<< "\n";
  os << "trigSourceVect = " << trigSourceVect
     << ", activeCFEBs = " << activeCFEBs << "\n";
  os << "bxnPreTrigger = " << bxnPreTrigger << "\n";
  os << "tmbMatch = " << tmbMatch << " alctOnly = " << alctOnly
     << " clctOnly = " << clctOnly
     << " alctMatchTime = " << alctMatchTime << "\n";
  os << "hs_thresh = " << hs_thresh << ", ds_thresh = " << ds_thresh
     << "\n";
  os << "clct0_key = " << clct0_key << " clct0_shape = " << clct0_shape
     << " clct0_quality = " << clct0_quality << "\n";
  os << "r_buf_nbusy = " << r_buf_nbusy << "\n";

  os << "..................CLCT....................." << "\n";
}


//unsigned short * data()  = 0;

