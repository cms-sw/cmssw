#include "EventFilter/CSCRawToDigi/interface/CSCTMBHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include <math.h>
#include <string.h> // memcpy

bool CSCTMBHeader::debug = false;

CSCTMBHeader::CSCTMBHeader() {
  bzero(this, sizeInWords()*2);
  nHeaderFrames = 26;
  e0bline = 0x6E0B;
  b0cline = 0x6B0C;
  nTBins = 7; 
  nCFEBs = 5;
}

void CSCTMBHeader::setEventInformation(const CSCDMBHeader & dmbHeader) {
  cscID = dmbHeader.dmbID();
  l1aNumber = dmbHeader.l1a();
  bxnCount = dmbHeader.bxn();
}


std::vector<CSCCLCTDigi> CSCTMBHeader::CLCTDigis() const {
  std::vector<CSCCLCTDigi> result;

  ///fill digis here
  /// for the zeroth clct:
  CSCCLCTDigi digi(clct0Valid(),clct0Quality(),clct0Shape(),clct0StripType(),clct0Bend(), 
		   clct0Key(),clct0CFEB(), clct0BXN(), 1);
  result.push_back(digi);

  /// for the first clct:
  digi = CSCCLCTDigi(clct1Valid(),clct1Quality(),clct1Shape(),clct1StripType(),clct1Bend(), 
		   clct1Key(),clct1CFEB(), clct1BXN(), 2);
  result.push_back(digi);
  
  return result;
}



std::ostream & operator<<(std::ostream & os, const CSCTMBHeader & hdr) {

  os << "...............TMB Header.................." << std::endl;
  os << std::hex << "BOC LINE " << hdr.b0cline << " EOB " << hdr.e0bline << std::endl;
  os << std::dec << "fifoMode = " << hdr.fifoMode << ", nTBins = " << hdr.nTBins << std::endl;
  os << "dumpCFEBs = " << hdr.dumpCFEBs << ", nHeaderFrames = " 
     << hdr.nHeaderFrames << std::endl;
  os << "boardID = " << hdr.boardID << ", cscID = " << hdr.cscID << std::endl;
  os << "l1aNumber = " << hdr.l1aNumber << ", bxnCount = " << hdr.bxnCount << std::endl;
  os << "preTrigTBins = " << hdr.preTrigTBins << ", nCFEBs = "<< hdr.nCFEBs<< std::endl;
  os << "trigSourceVect = " << hdr.trigSourceVect 
     << ", activeCFEBs = " << hdr.activeCFEBs << std::endl;
  os << "bxnPreTrigger = " << hdr.bxnPreTrigger << std::endl;
  os << "tmbMatch = " << hdr.tmbMatch << " alctOnly = " << hdr.alctOnly << " clctOnly = " << hdr.clctOnly << std::endl ;


  os << "..................CLCT....................." << std::endl;

  return os;

}
