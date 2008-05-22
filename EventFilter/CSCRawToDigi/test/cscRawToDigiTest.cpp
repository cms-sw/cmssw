#include "EventFilter/CSCRawToDigi/interface/CSCTMBHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCALCTHeader.h"

int main()
{
  CSCTMBHeader::selfTest();
  CSCALCTHeader::selfTest();
}

