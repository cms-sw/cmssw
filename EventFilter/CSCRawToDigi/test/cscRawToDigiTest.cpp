#include "EventFilter/CSCRawToDigi/interface/CSCTMBHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCTMBData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCALCTHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCEventData.h"

int main()
{
  CSCTMBHeader::selfTest();
  CSCALCTHeader::selfTest();
  CSCTMBData::selfTest();
  CSCEventData::selfTest();
}

