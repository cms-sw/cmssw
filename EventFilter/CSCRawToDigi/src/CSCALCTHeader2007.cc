#include "EventFilter/CSCRawToDigi/interface/CSCALCTHeader2007.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader.h"

CSCALCT::CSCALCT()  {
  bzero(this, 2); ///size of ALCT = 2bytes
}


CSCALCT::CSCALCT(const CSCALCTDigi & alctDigi):
    valid(alctDigi.isValid()),
    quality(alctDigi.getQuality()),
    accel(alctDigi.getAccelerator()),
    pattern(alctDigi.getCollisionB()),
    keyWire(alctDigi.getKeyWG()),
    reserved(0)
{
}

#include <iostream>
CSCALCTHeader2007::CSCALCTHeader2007()
{
  bzero(this,  sizeInWords()*2); ///size of 2007 header w/o variable parts = 16 bytes
  flag1 = 0xDB0A;
  reserved1 = reserved2 = reserved3 = 0xD;
  rawBins = 16;
  lctBins = 8;
}

CSCALCTHeader2007::CSCALCTHeader2007(int chamberType)  {
  bzero(this,  sizeInWords()*2); ///size of 2007 header w/o variable parts = 16 bytes
  // things that depend on chamber type
  int boardTypes[11] = {0, 2, 2, 3, 1, 6, 3, 5, 3, 5, 3};
  flag1 = 0xDB0A;
  reserved1 = reserved2 = reserved3 = 0xD;
  boardType = boardTypes[chamberType];
  //FIXME how do BXes work?  Dump other raw data
  // shows rawBins=16lctBins=8 or rawbins=0, lctBins=1
  rawBins = 16;
  lctBins = 8;
}

void CSCALCTHeader2007::setEventInformation(const CSCDMBHeader & dmb)
{
 l1aCounter = dmb.l1a24() & 0xFFF;
 bxnCount = dmb.bxn12();
}

