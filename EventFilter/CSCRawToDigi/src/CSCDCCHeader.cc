#include "EventFilter/CSCRawToDigi/interface/CSCDCCHeader.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"


CSCDCCHeader::CSCDCCHeader(int bx, int l1a, int sourceId)
{
  word[0] = 0x510000000000005FLL;
  word[1] = 0xD900000000000097LL;
  FEDHeader::set(reinterpret_cast<unsigned char *>(data()), 1, l1a, bx, sourceId);
}


CSCDCCHeader::CSCDCCHeader() 
{
  word[0] = 0x510000000000005FLL;
  word[1] = 0xD900000000000097LL;
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


std::ostream & operator<<(std::ostream & os, const CSCDCCHeader & hdr) 
{
  os << "DCC Header" << std::endl;
  return os;
}

