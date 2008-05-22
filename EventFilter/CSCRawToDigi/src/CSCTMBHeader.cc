#include "EventFilter/CSCRawToDigi/interface/CSCTMBHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include <math.h>
#include <string.h> // memcpy

bool CSCTMBHeader::debug = false;
short unsigned int CSCTMBHeader::firmwareVersion=2006;

CSCTMBHeader::CSCTMBHeader() {
  firmwareVersion=2006;
  header2006.nHeaderFrames = 26;
  header2006.e0bline = 0x6E0B;
  header2006.b0cline = 0x6B0C;
  header2006.nTBins = 7; 
  header2006.nCFEBs = 5;
}

void CSCTMBHeader::setEventInformation(const CSCDMBHeader & dmbHeader) {
  header2006.cscID = dmbHeader.dmbID();
  header2006.l1aNumber = dmbHeader.l1a();
  header2006.bxnCount = dmbHeader.bxn();
}

CSCTMBHeader::CSCTMBHeader(const CSCTMBStatusDigi & digi) {
  CSCTMBHeader(digi.header());
}

CSCTMBHeader::CSCTMBHeader(const unsigned short * buf) {
  ///first determine the format
  if (buf[0]==0xDB0C) {
    firmwareVersion=2007;
  }
  else if (buf[0]==0x6B0C) {
    firmwareVersion=2006;
  }
  else {
    edm::LogError("CSCTMBHeader") <<"failed to determine TMB firmware version!!";
  }
    
  ///now fill data
  switch (firmwareVersion) {
  case 2006:
    memcpy(&header2006, buf, header2006.sizeInWords()*2);    
    break;
  case 2007:
    memcpy(&header2007, buf, header2007.sizeInWords()*2);
    break;
  default:
    edm::LogError("CSCTMBHeader")
      <<"coundn't construct: TMB firmware version is bad/not defined!";
    break;
  }
}	
    

std::vector<CSCCLCTDigi> CSCTMBHeader::CLCTDigis(uint32_t idlayer) const {
  std::vector<CSCCLCTDigi> result;

  switch (firmwareVersion) {
  case 2006: {
    ///fill digis here
    /// for the zeroth clct:
    int shape=0;
    int type=0;
  
    if ( header2006.firmRevCode < 3769 ) { //3769 is may 25 2007 - date of firmware with halfstrip only patterns 
    shape = header2006.clct0_shape;
    type  = header2006.clct0_strip_type;
    }else {//new firmware only halfstrip pattern => stripType==1 and shape is 4 bits 
      shape = ( header2006.clct0_strip_type<<3)+header2006.clct0_shape;
      type = 1;
    }
    int strip = header2006.clct0_key;
    int cfeb = (header2006.clct0_cfeb_low)|(header2006.clct0_cfeb_high<<1);
    correctCLCTNumbering(idlayer, strip, cfeb);

    CSCCLCTDigi digi0(header2006.clct0_valid, header2006.clct0_quality, shape, type, header2006.clct0_bend, 
		      strip, cfeb, header2006.clct0_bxn, 1);
    digi0.setFullBX(header2006.bxnPreTrigger);
    result.push_back(digi0);
    
    /// for the first clct:
    if ( header2006.firmRevCode < 3769 ) { 
      shape = header2006.clct1_shape;
      type  = header2006.clct1_strip_type;;
    } else {
      shape = (header2006.clct1_strip_type<<3)+header2006.clct1_shape;
      type = 1;
    }

    strip = header2006.clct1_key;
    cfeb = (header2006.clct1_cfeb_low)|(header2006.clct1_cfeb_high<<1);
    correctCLCTNumbering(idlayer, strip, cfeb);

    CSCCLCTDigi digi1(header2006.clct1_valid, header2006.clct1_quality, shape, type, header2006.clct1_bend,
		      strip, cfeb, header2006.clct1_bxn, 2);
    digi1.setFullBX(header2006.bxnPreTrigger);
    result.push_back(digi1);
    break;
  }
  case 2007: {
    int strip = header2007.clct0_key;
    int cfeb = (header2007.clct0_cfeb_low)|(header2007.clct0_cfeb_high<<1);
    correctCLCTNumbering(idlayer, strip, cfeb);

    CSCCLCTDigi digi0(header2007.clct0_valid, header2007.clct0_quality, header2007.clct0_shape, 1, 
		      header2007.clct0_bend, strip, cfeb, header2007.clct0_bxn, 1);
    digi0.setFullBX(header2007.bxnPreTrigger);
    result.push_back(digi0);
    strip = header2007.clct1_key;
    cfeb = (header2007.clct1_cfeb_low)|(header2007.clct1_cfeb_high<<1);
    correctCLCTNumbering(idlayer, strip, cfeb);

    CSCCLCTDigi digi1(header2007.clct1_valid, header2007.clct1_quality, header2007.clct1_shape, 1,
                      header2007.clct1_bend, strip, cfeb, header2007.clct1_bxn, 2);
    digi1.setFullBX(header2007.bxnPreTrigger);
    result.push_back(digi1);
    break;
  }
  default:
    edm::LogError("CSCTMBHeader")
      <<"Empty Digis: TMB firmware version is bad/not defined!"; 
    break;
  }
  return result;
}


void CSCTMBHeader::correctCLCTNumbering(uint32_t idlayer, int & strip, int & cfeb) const
{
  bool me1a = (CSCDetId::station(idlayer)==1) && (CSCDetId::ring(idlayer)==4);
  bool zplus = (CSCDetId::endcap(idlayer) == 1);
  bool me1b = (CSCDetId::station(idlayer)==1) && (CSCDetId::ring(idlayer)==1);

  if ( me1a ) cfeb = 0; // reset cfeb 4 to 0
  if ( me1a && zplus ) { strip = 31-strip; } // 0-31 -> 31-0
  if ( me1b && !zplus) { cfeb = 4 - cfeb; strip = 31 - strip;} // 0-127 -> 127-0 ...
}

void CSCTMBHeader::correctCorrLCTNumbering(uint32_t idlayer, int & strip) const
{
  bool me1a = (CSCDetId::station(idlayer)==1) && (CSCDetId::ring(idlayer)==4);
  bool zplus = (CSCDetId::endcap(idlayer) == 1);
  bool me1b = (CSCDetId::station(idlayer)==1) && (CSCDetId::ring(idlayer)==1);

  if ( me1a ) strip = strip%128; // reset strip 128-159 -> 0-31
  if ( me1a && zplus ) { strip = 31-strip; } // 0-31 -> 31-0
  if ( me1b && !zplus) { strip = 127 - strip;} // 0-127 -> 127-0 ...
}


std::vector<CSCCorrelatedLCTDigi> CSCTMBHeader::CorrelatedLCTDigis(uint32_t idlayer) const {
  std::vector<CSCCorrelatedLCTDigi> result;  

  switch (firmwareVersion) {
  case 2006: {
    /// for the zeroth MPC word:
    int strip = header2006.MPC_Muon0_halfstrip_clct_pattern;//this goes from 0-159
    correctCorrLCTNumbering(idlayer, strip);
    CSCCorrelatedLCTDigi digi(1, header2006.MPC_Muon0_vpf_, header2006.MPC_Muon0_quality_, 
			      header2006.MPC_Muon0_wire_, strip, header2006.MPC_Muon0_clct_pattern_,
			      header2006.MPC_Muon0_bend_, header2006.MPC_Muon0_bx_, 0, 
			      header2006.MPC_Muon0_bc0_, header2006.MPC_Muon0_SyncErr_, 
			      header2006.MPC_Muon0_cscid_low | (header2006.MPC_Muon0_cscid_bit4<<3) );
    result.push_back(digi);  
    /// for the first MPC word:
    strip = header2006.MPC_Muon1_halfstrip_clct_pattern;//this goes from 0-159
    correctCorrLCTNumbering(idlayer, strip);
    digi = CSCCorrelatedLCTDigi(2, header2006.MPC_Muon1_vpf_, header2006.MPC_Muon1_quality_, 
				header2006.MPC_Muon1_wire_, strip, header2006.MPC_Muon1_clct_pattern_,
				header2006.MPC_Muon1_bend_, header2006.MPC_Muon1_bx_, 0, 
				header2006.MPC_Muon1_bc0_, header2006.MPC_Muon1_SyncErr_,
				header2006.MPC_Muon1_cscid_low | (header2006.MPC_Muon1_cscid_bit4<<3) ); 
    result.push_back(digi);
    break;
  }
  case 2007: {
    /// for the zeroth MPC word:
    int strip = header2007.MPC_Muon0_halfstrip_clct_pattern;//this goes from 0-159
    correctCorrLCTNumbering(idlayer, strip);
    CSCCorrelatedLCTDigi digi(1, header2007.MPC_Muon0_vpf_, header2007.MPC_Muon0_quality_,
                              header2007.MPC_Muon0_wire_, strip, header2007.MPC_Muon0_clct_pattern_, 
			      header2007.MPC_Muon0_bend_, header2007.MPC_Muon0_bx_, 0, 
			      header2007.MPC_Muon0_bc0_, header2007.MPC_Muon0_SyncErr_,
                              header2007.MPC_Muon0_cscid_low | (header2007.MPC_Muon0_cscid_bit4<<3));
    result.push_back(digi);
    /// for the first MPC word:
    strip = header2007.MPC_Muon1_halfstrip_clct_pattern;//this goes from 0-159
    correctCorrLCTNumbering(idlayer, strip);
    digi = CSCCorrelatedLCTDigi(2, header2007.MPC_Muon1_vpf_, header2007.MPC_Muon1_quality_,
                                header2007.MPC_Muon1_wire_, strip, header2007.MPC_Muon1_clct_pattern_, 
				header2007.MPC_Muon1_bend_, header2007.MPC_Muon1_bx_, 0, 
				header2007.MPC_Muon1_bc0_, header2007.MPC_Muon1_SyncErr_,
                                header2007.MPC_Muon1_cscid_low | (header2007.MPC_Muon1_cscid_bit4<<3));
    result.push_back(digi);
    break;
  }
  default:
    edm::LogError("CSCTMBHeader")
      <<"Empty CorrDigis: TMB firmware version is bad/not defined!";
    break;
  }
  return result;
}



void CSCTMBHeader2007::addALCT0(const CSCALCTDigi & digi)
{
  alct0Valid = digi.isValid();
  alct0Quality = digi.getQuality();
  alct0Amu = digi.getAccelerator();
  alct0Key = digi.getKeyWG();
  reserved2 = 0;
  flag28 = 0;
}

void CSCTMBHeader2007::addALCT1(const CSCALCTDigi & digi)
{
  alct1Valid = digi.isValid();
  alct1Quality = digi.getQuality();
  alct1Amu = digi.getAccelerator();
  alct1Key = digi.getKeyWG();
  reserved3 = 0;
  flag29 = 0;
}


void CSCTMBHeader::addCLCT0(const CSCCLCTDigi & digi)
{
  if(firmwareVersion == 2006)
    addCLCT0(digi, header2006);
  else
    addCLCT0(digi, header2007);
}

void CSCTMBHeader::addCLCT1(const CSCCLCTDigi & digi)
{
  if(firmwareVersion == 2006)
    addCLCT1(digi, header2006);
  else
    addCLCT1(digi, header2007);
}

void CSCTMBHeader::addALCT0(const CSCALCTDigi & digi)
{
  if(firmwareVersion == 2006) {
    //addALCT0(digi, header2006);
  }
  else
    header2007.addALCT0(digi);
}

void CSCTMBHeader::addALCT1(const CSCALCTDigi & digi)
{
  if(firmwareVersion == 2006) {
    //addALCT0(digi, header2006);
  }
  else
    header2007.addALCT1(digi);
}

void CSCTMBHeader::addCorrelatedLCT0(const CSCCorrelatedLCTDigi & digi)
{
  if(firmwareVersion == 2006)
    addCorrelatedLCT0(digi, header2006);
  else
    addCorrelatedLCT0(digi, header2007);
}

void CSCTMBHeader::addCorrelatedLCT1(const CSCCorrelatedLCTDigi & digi)
{
  if(firmwareVersion == 2006)
    addCorrelatedLCT1(digi, header2006);
  else
    addCorrelatedLCT1(digi, header2007);
}


void CSCTMBHeader::selfTest()
{
  // tests packing and unpacking
  for(int station = 1; station <= 4; ++station)
  {
    CSCDetId detId(1, station, 1, 1, 0);
    CSCCLCTDigi clct0(true, 1, 2, 1, 2, 30, 3, 6, 1);
    CSCCLCTDigi clct1(true, 1, 2, 3, 4, 31, 2, 6, 1);

    CSCCorrelatedLCTDigi lct0(1, 1, 2, 10, 8, 5, 1, 6, 0, 0, 0, 0);
    CSCCorrelatedLCTDigi lct1(1, 1, 2, 20, 15, 5, 1, 6, 0, 0, 0, 0);

    CSCTMBHeader tmbHeader;

    tmbHeader.addCLCT0(clct0);
    tmbHeader.addCLCT1(clct1);
    tmbHeader.addCorrelatedLCT0(lct0);
    tmbHeader.addCorrelatedLCT1(lct1);

    std::vector<CSCCLCTDigi> clcts = tmbHeader.CLCTDigis(detId.rawId());
    assert(clcts[0] == clct0);
    assert(clcts[0] == clct1);

    std::vector<CSCCorrelatedLCTDigi> lcts = tmbHeader.CorrelatedLCTDigis(detId.rawId());
    assert(lcts[0] == lct0);
    assert(lcts[0] == lct1);

  }

}


std::ostream & operator<<(std::ostream & os, const CSCTMBHeader & hdr) {
  os << "...............TMB Header.................." << std::endl;
  os << std::hex << "BOC LINE " << hdr.header2006.b0cline << " EOB " << hdr.header2006.e0bline << std::endl;
  os << std::dec << "fifoMode = " << hdr.header2006.fifoMode 
     << ", nTBins = " << hdr.header2006.nTBins << std::endl;
  os << "dumpCFEBs = " << hdr.header2006.dumpCFEBs << ", nHeaderFrames = "
     << hdr.header2006.nHeaderFrames << std::endl;
  os << "boardID = " << hdr.header2006.boardID << ", cscID = " << hdr.header2006.cscID << std::endl;
  os << "l1aNumber = " << hdr.header2006.l1aNumber << ", bxnCount = " << hdr.header2006.bxnCount << std::endl;
  os << "preTrigTBins = " << hdr.header2006.preTrigTBins << ", nCFEBs = "<< hdr.header2006.nCFEBs<< std::endl;
  os << "trigSourceVect = " << hdr.header2006.trigSourceVect
     << ", activeCFEBs = " << hdr.header2006.activeCFEBs << std::endl;
  os << "bxnPreTrigger = " << hdr.header2006.bxnPreTrigger << std::endl;
  os << "tmbMatch = " << hdr.header2006.tmbMatch << " alctOnly = " << hdr.header2006.alctOnly
     << " clctOnly = " << hdr.header2006.clctOnly
     << " alctMatchTime = " << hdr.header2006.alctMatchTime << std::endl;
  os << "hs_thresh = " << hdr.header2006.hs_thresh << ", ds_thresh = " << hdr.header2006.ds_thresh
     << std::endl;
  os << "clct0_key = " << hdr.header2006.clct0_key << " clct0_shape = " << hdr.header2006.clct0_shape
     << " clct0_quality = " << hdr.header2006.clct0_quality << std::endl;
  os << "r_buf_nbusy = " << hdr.header2006.r_buf_nbusy << std::endl;

  os << "..................CLCT....................." << std::endl;
  return os;
}
