#include "EventFilter/CSCRawToDigi/interface/CSCTMBTrailer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <cassert>

CSCTMBTrailer::CSCTMBTrailer(int wordCount, int firmwareVersion) 
: theFirmwareVersion(firmwareVersion)
{
  //FIXME do firmware version
  theData[0] = 0x6e0c;
  // all the necessary lines from this thing first
  wordCount += 5;
  // see if we need thePadding to make a multiple of 4
  thePadding = 0;

  if(wordCount%4==2) 
    {
      theData[1] = 0x2AAA;
      theData[2] = 0x5555;
      thePadding = 2;
      wordCount += thePadding;
    }
  // the next four words start with 11011, or a D
  for(int i = 1; i < 5; ++i) 
    {
      theData[i+thePadding] = 0xD800;
    }
  theData[de0fOffset()] = 0xde0f;
  // word count excludes the trailer
  theData[4+thePadding] |= wordCount;
}


CSCTMBTrailer::CSCTMBTrailer(unsigned short * buf, unsigned short int firmwareVersion) 
: theFirmwareVersion(firmwareVersion)
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
    // =VB= check for 1st word to be 0xDE0F, then check 3rd 
    // to handle freaky cases of double 0xDE0F signatures in trailer 
    thePadding = (theData[1] == 0xde0f ? 0 : (theData[3] == 0xde0f ? 2 : 0)); 
//    thePadding = (theData[3] == 0xde0f ? 2 : 0);
    break;
  default: 
    edm::LogError("CSCTMBTrailer|CSCRawToDigi")
      <<"failed to contruct: firmware version is bad/not defined!";
  }
}

unsigned int CSCTMBTrailer::crc22() const 
{  
  return (theData[crcOffset()] & 0x07ff) +
            ((theData[crcOffset()+1] & 0x07ff) << 11);
}


void CSCTMBTrailer::setCRC(int crc) 
{
  theData[crcOffset()] |= (crc & 0x07ff);
  theData[crcOffset()+1] |= ((crc>>11) & 0x07ff);
}


int CSCTMBTrailer::wordCount() const {return theData[4+thePadding] & 0x7ff;}



void CSCTMBTrailer::selfTest()
{
  CSCTMBTrailer trailer(104, 2006);
  unsigned int crc = 0xb00b1;
  trailer.setCRC(crc);
  assert(trailer.crc22() == 0xb00b1);

  CSCTMBTrailer trailer2(104, 2007);
  crc = 0xb00b1;
  trailer2.setCRC(crc);
  assert(trailer2.crc22() == 0xb00b1);

}

