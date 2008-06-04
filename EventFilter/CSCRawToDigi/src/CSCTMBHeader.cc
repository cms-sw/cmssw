#include "EventFilter/CSCRawToDigi/interface/CSCTMBHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader.h"
#include "EventFilter/CSCRawToDigi/src/cscPackerCompare.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
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
    

std::vector<CSCCLCTDigi> CSCTMBHeader::CLCTDigis(uint32_t idlayer) {
  std::vector<CSCCLCTDigi> result;
  theChamberId = CSCDetId(idlayer);


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
    int bend = header2006.clct0_bend;
    offlineStripNumbering(firmwareVersion, strip, cfeb, shape, bend);

    CSCCLCTDigi digi0(header2006.clct0_valid, header2006.clct0_quality, shape,
		      type, bend, strip, cfeb, header2006.clct0_bxn, 1);
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
    bend = header2006.clct1_bend;
    offlineStripNumbering(firmwareVersion, strip, cfeb, shape, bend);

    CSCCLCTDigi digi1(header2006.clct1_valid, header2006.clct1_quality, shape,
		      type, bend, strip, cfeb, header2006.clct1_bxn, 2);
    digi1.setFullBX(header2006.bxnPreTrigger);
    result.push_back(digi1);
    break;
  }
  case 2007: {
    int strip   = header2007.clct0_key;
    int cfeb    = (header2007.clct0_cfeb_low)|(header2007.clct0_cfeb_high<<1);
    int pattern = header2007.clct0_shape;
    int bend    = header2007.clct0_bend;
    offlineStripNumbering(firmwareVersion, strip, cfeb, pattern, bend);

    CSCCLCTDigi digi0(header2007.clct0_valid, header2007.clct0_quality,
		      pattern, 1, bend, strip, cfeb, header2007.clct0_bxn, 1);
    digi0.setFullBX(header2007.bxnPreTrigger);

    strip = header2007.clct1_key;
    cfeb = (header2007.clct1_cfeb_low)|(header2007.clct1_cfeb_high<<1);
    pattern = header2007.clct1_shape;
    bend    = header2007.clct1_bend;
    offlineStripNumbering(firmwareVersion, strip, cfeb, pattern, bend);

    CSCCLCTDigi digi1(header2007.clct1_valid, header2007.clct1_quality,
		      pattern, 1, bend, strip, cfeb, header2007.clct1_bxn, 2);
    digi1.setFullBX(header2007.bxnPreTrigger);

    if (digi0.isValid() && digi1.isValid()) swapCLCTs(digi0, digi1);

    result.push_back(digi0);
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


void CSCTMBHeader::offlineStripNumbering(const int firmwareVersion,
					 int & strip, int & cfeb,
					 int & pattern, int & bend) const
{
  bool me11 = (theChamberId.station() == 1 && 
	       (theChamberId.ring() == 1 || theChamberId.ring() == 4));
  if (!me11) return;

  // SV, 30/05/2008: Rely directly on CFEB number instead of DetId
  // ring to tell ME1/a from ME1/b.  For the time being (May 2008),
  // the CLCT trigger logic combines comparators in ME1/a with
  // comparators in ME1/b and performs seamless search for CLCTs in
  // all 5 CFEBs.  In order to reproduce this in the trigger emulator,
  // keep both comparators and CLCTs in the same DetId (ring=1)
  // regardless of whether they belong to ME1/a or ME1/b.  So this
  // method is called with theChamberId.ring()=1 for both ME1/a and
  // ME1/b.
  bool me1a = (cfeb == 4);
  bool me1b = (cfeb != 4);
  bool zplus = (theChamberId.endcap() == 1);

  // Keep CFEB=4 for now.
  // if ( me1a ) cfeb = 0; // reset cfeb 4 to 0
  if ( me1a && zplus  ) { strip = 31-strip; } // 0-31 -> 31-0
  if ( me1b && !zplus ) { cfeb = 3-cfeb; strip = 31-strip;} // 0-127 -> 127-0
  if ((me1a && zplus) || (me1b && !zplus)) {
    switch (firmwareVersion) {
    case 2006: 
      if (pattern > 0 && pattern < 6) {
	if (pattern % 2 == 0) {pattern--; bend++;}
	else                  {pattern++; bend--;}
      }
      break;
    case 2007: 
      if (pattern > 1 && pattern < 10) {
	if (pattern % 2 == 0) {pattern++; bend++;}
	else                  {pattern--; bend--;}
      }
      break;
    default:
      break;
    }
  }
}

void CSCTMBHeader::swapCLCTs(CSCCLCTDigi& digi1, CSCCLCTDigi& digi2)
{
  bool me11 = (theChamberId.station() == 1 && 
	       (theChamberId.ring() == 1 || theChamberId.ring() == 4));
  if (!me11) return;

  int cfeb1 = digi1.getCFEB();
  int cfeb2 = digi2.getCFEB();
  if (cfeb1 != cfeb2) return;

  bool me1a = (cfeb1 == 4);
  bool me1b = (cfeb1 != 4);
  bool zplus = (theChamberId.endcap() == 1);

  if ( (me1a && zplus) || (me1b && !zplus)) {
    // Swap CLCTs if they have the same quality and pattern # (priority
    // has to be given to the lower key).
    if (digi1.getQuality() == digi2.getQuality() &&
	digi1.getPattern() == digi2.getPattern()) {
      CSCCLCTDigi temp = digi1;
      digi1 = digi2;
      digi2 = temp;

      // Also re-number them.
      digi1.setTrknmb(1);
      digi2.setTrknmb(2);
    }
  }
}

void CSCTMBHeader::hardwareStripNumbering(int & strip, int & cfeb) const
{
  bool me1a = (theChamberId.station()==1 && theChamberId.ring() == 4);
  bool zplus = (theChamberId.endcap() == 1);
  bool me1b = (theChamberId.station()==1 && theChamberId.ring() == 1);

  if ( me1a ) cfeb = 4; 
  if ( me1a && zplus ) { strip = 31-strip; } // 0-31 -> 31-0
  if ( me1b && !zplus) { cfeb = 3 - cfeb; strip = 31 - strip;} // 0-127 -> 127-0 ...
}


void CSCTMBHeader::offlineHalfStripNumbering(int & strip) const
{
  bool me11 = (theChamberId.station() == 1 && 
	       (theChamberId.ring() == 1 || theChamberId.ring() == 4));
  if (!me11) return;

  // SV, 30/05/2008: Rely directly on half-strip number to tell ME1/a from
  // ME1/b; see comments to offlineStripNumbering().
  bool me1a = (strip >= 128);
  bool me1b = (strip <  128);
  bool zplus = (theChamberId.endcap() == 1);

  //if ( me1a ) strip = strip%128; // reset strip 128-159 -> 0-31
  //if ( me1a && zplus ) { strip = 31-strip; } // 0-31 -> 31-0
  if ( me1a && zplus ) { strip = 128 + (159 - strip); } // 128-159 -> 159-128
  if ( me1b && !zplus) { strip = 127 - strip; } // 0-127 -> 127-0 ...
}


void CSCTMBHeader::hardwareHalfStripNumbering(int & strip) const
{
  bool me1a = (theChamberId.station()==1 && theChamberId.ring() == 4);
  bool zplus = (theChamberId.endcap() == 1);
  bool me1b = (theChamberId.station()==1 && theChamberId.ring() == 1);

  if ( me1a ) strip += 128; // reset strip  0-31 -> 128-159
  if ( me1a && zplus ) { strip = 31-strip; } // 0-31 -> 31-0
  if ( me1b && !zplus) { strip = 127 - strip;} // 0-127 -> 127-0 ...
}



std::vector<CSCCorrelatedLCTDigi> CSCTMBHeader::CorrelatedLCTDigis(uint32_t idlayer) const {
  theChamberId = CSCDetId(idlayer);

  std::vector<CSCCorrelatedLCTDigi> result;  

  switch (firmwareVersion) {
  case 2006: {
    /// for the zeroth MPC word:
    int strip = header2006.MPC_Muon0_halfstrip_clct_pattern;//this goes from 0-159
    offlineHalfStripNumbering(strip);
    CSCCorrelatedLCTDigi digi(1, header2006.MPC_Muon0_vpf_, header2006.MPC_Muon0_quality_, 
			      header2006.MPC_Muon0_wire_, strip, header2006.MPC_Muon0_clct_pattern_,
			      header2006.MPC_Muon0_bend_, header2006.MPC_Muon0_bx_, 0, 
			      header2006.MPC_Muon0_bc0_, header2006.MPC_Muon0_SyncErr_, 
			      header2006.MPC_Muon0_cscid_low | (header2006.MPC_Muon0_cscid_bit4<<3) );
    result.push_back(digi);  
    /// for the first MPC word:
    strip = header2006.MPC_Muon1_halfstrip_clct_pattern;//this goes from 0-159
    offlineHalfStripNumbering(strip);
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
    offlineHalfStripNumbering(strip);
    CSCCorrelatedLCTDigi digi(1, header2007.MPC_Muon0_vpf_, header2007.MPC_Muon0_quality_,
                              header2007.MPC_Muon0_wire_, strip, header2007.MPC_Muon0_clct_pattern_, 
			      header2007.MPC_Muon0_bend_, header2007.MPC_Muon0_bx_, 0, 
			      header2007.MPC_Muon0_bc0_, header2007.MPC_Muon0_SyncErr_,
                              header2007.MPC_Muon0_cscid_low | (header2007.MPC_Muon0_cscid_bit4<<3));
    result.push_back(digi);
    /// for the first MPC word:
    strip = header2007.MPC_Muon1_halfstrip_clct_pattern;//this goes from 0-159
    offlineHalfStripNumbering(strip);
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
    for(int iendcap = 0; iendcap <= 1; ++iendcap) 
    {
      CSCDetId detId(iendcap, station, 1, 1, 0);
      // the next-to-last is the BX, which only gets
      // saved in two bits... I guess the bxnPreTrigger is involved?
      CSCCLCTDigi clct0(1, 1, 2, 0, 0, 30, 3, 6, 1);
      CSCCLCTDigi clct1(1, 1, 2, 1, 1, 31, 2, 7, 1);

      CSCCorrelatedLCTDigi lct0(1, 1, 2, 10, 8, 5, 1, 6, 0, 0, 0, 0);
      CSCCorrelatedLCTDigi lct1(1, 1, 2, 20, 15, 5, 1, 6, 0, 0, 0, 0);

      CSCTMBHeader tmbHeader;

      tmbHeader.addCLCT0(clct0);
      tmbHeader.addCLCT1(clct1);
      tmbHeader.addCorrelatedLCT0(lct0);
      tmbHeader.addCorrelatedLCT1(lct1);

      std::vector<CSCCLCTDigi> clcts = tmbHeader.CLCTDigis(detId.rawId());
      assert(cscPackerCompare(clcts[0],clct0));
      assert(cscPackerCompare(clcts[0],clct1));

      std::vector<CSCCorrelatedLCTDigi> lcts = tmbHeader.CorrelatedLCTDigis(detId.rawId());
      assert(cscPackerCompare(lcts[0], lct0));
      assert(cscPackerCompare(lcts[0], lct1));
    }
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
