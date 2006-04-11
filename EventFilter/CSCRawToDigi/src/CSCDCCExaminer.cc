/// CSCDCCExaminer.cc
/// 04-07-06
/// A.Tumanov


#include "EventFilter/CSCRawToDigi/interface/CSCDCCExaminer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

bool CSCDCCExaminer::debug = false;


bool CSCDCCExaminer::examine(unsigned short int * buf, unsigned short int length) {
  bool result=false;

  ///this is just a rudimentary check for the DDU trailer to be near the DCC trailer
  unsigned short int start=length/2-20;
  result = (buf[start+3]==0x8000)&&(buf[start+2]==0xffff)&&(buf[start+1]==0x8000)&&(buf[start]==0x8000);
  return result;
}


