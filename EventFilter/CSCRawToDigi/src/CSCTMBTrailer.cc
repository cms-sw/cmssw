#include "EventFilter/CSCRawToDigi/interface/CSCTMBTrailer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


CSCTMBTrailer::CSCTMBTrailer(int wordCount) 
{
  theData[0] = 0x6e0c;
  // see if we need thePadding to make a multiple of 4
  thePadding = 0;
  if(wordCount%4==2) 
    {
      theData[1] = 0x2AAA;
      theData[2] = 0x5555;
      thePadding = 2;
    }
  // the next four words start with 11011, or a D
  for(int i = 1; i < 5; ++i) 
    {
      theData[i+thePadding] = (0x1B << 11);
    }
  theData[3+thePadding] = 0xde0f;
  theData[4+thePadding] |= wordCount+ thePadding;
}


CSCTMBTrailer::CSCTMBTrailer(unsigned short * buf, unsigned short int firmwareVersion) 
{
  // take a little too much, maybe
  memcpy(theData, buf, 14);
  switch (firmwareVersion){
  case 2006:
    // if there's padding, there'll be a de0f in the 6th word.
    // If not, you'll be in CFEB-land, where they won't be de0f.
    thePadding = (theData[5] == 0xde0f ? 2 : 0);
    break;
  case 2007:
    ///in 2007 format de0f line moved
    thePadding = (theData[3] == 0xde0f ? 2 : 0);
    break;
  default: 
    edm::LogError("CSCTMBTrailer|CSCRawToDigi")
      <<"failed to contruct: firmware version is bad/not defined!";
  }
}

int CSCTMBTrailer::crc22() const {return theData[1+thePadding] & 0x7fff + ((theData[2+thePadding] & 0x7fff) << 11);}

int CSCTMBTrailer::wordCount() const {return theData[4+thePadding] & 0x7ff;}
void CSCTMBTrailer::setWordCount(int words) {theData[4+thePadding] |= (words&0x7ff);}

