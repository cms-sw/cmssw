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

  //fill digis here


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
