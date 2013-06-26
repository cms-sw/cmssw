#include "EventFilter/CSCRawToDigi/interface/CSCTMBHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCTMBData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCALCTHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCAnodeData2006.h"
#include "EventFilter/CSCRawToDigi/interface/CSCAnodeData2007.h"

int main()
{
  CSCTMBHeader::selfTest();
  CSCALCTHeader::selfTest(2007);
  CSCALCTHeader::selfTest(2006);
  CSCTMBData::selfTest();
  CSCTMBTrailer::selfTest();
  //CSCEventData::selfTest();
  CSCCLCTData::selfTest();

  CSCAnodeData2006::selfTest();
  CSCAnodeData2007::selfTest();
}

