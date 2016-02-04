#include "EventFilter/CSCRawToDigi/interface/CSCDCCHeader.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include <iostream>
#include <cassert>
#include <cstring>

CSCDCCHeader::CSCDCCHeader(int bx, int l1a, int sourceId, int version)
{
  word[0] = 0x5100000000000008LL;
  word[1] = 0xD900000000000000LL;
  /// =VB= Should pass true as last parameter for FEDHeader::set() method to construct correct data
  FEDHeader::set(reinterpret_cast<unsigned char *>(data()), 1, l1a, bx, sourceId, version, true);
}


CSCDCCHeader::CSCDCCHeader() 
{
  word[0] = 0x5100000000000008LL;
  word[1] = 0xD900000000000000LL;
}

CSCDCCHeader::CSCDCCHeader(const CSCDCCStatusDigi & digi)
{
  memcpy(this, digi.header(), sizeInWords()*2);
}



int CSCDCCHeader::getCDFEventNumber() const 
{ 
  return ((word[0]>>32)&0x00FFFFFF);
}

int CSCDCCHeader::getCDFBunchCounter() const 
{ 
  return ((word[0]>>20)&0xFFF);
}
int CSCDCCHeader::getCDFSourceId() const 
{ 
  return ((word[0]>>8)&0xFFF);
}
int CSCDCCHeader::getCDFFOV() const 
{ 
  return ((word[0]>>4)&0xF);
}
int CSCDCCHeader::getCDFEventType() const 
{ 
  return ((word[0]>>56)&0xF);
}


void CSCDCCHeader::setDAV(int slot)
{
  /* Bits 7,6,5,4,2 to indicate available DDU. 
     For slink0, the DDU slots are 5, 12, 4, 13, 3 (same as Fifo_in_use[4:0]); 
     for slink1, the DDU slots are 9, 7, 10, 6, 11
  */
  assert(slot>=3 && slot <= 13);
  int bit[] = {0, 0, 0, 2, 5, 7, 4, 6, 0, 7, 5, 2, 6, 4};
  word[0] |= 1 << bit[slot];
}

std::ostream & operator<<(std::ostream & os, const CSCDCCHeader & hdr) 
{
  os << "DCC Header" << std::endl;
  return os;
}

