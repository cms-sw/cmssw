#include "EventFilter/CSCRawToDigi/interface/CSCDCCHeader.h"
#include <iostream>

CSCDCCHeader::CSCDCCHeader(int bx, int l1a, int sourceId) {
  bzero(this, sizeInWords()*2);
  dcc_code1 = 0xD9;
  dcc_code2 = 0x97;
  BX_id = bx;
  LV1_id = l1a;
  Source_id = sourceId;
}


CSCDCCHeader::CSCDCCHeader() {
  bzero(this, sizeInWords()*2);
  dcc_code1 = 0xD9;
  dcc_code2 = 0x97;
}


int CSCDCCHeader::getCDFEventNumber() const { 
  return ((word[0]>>32)&0x00FFFFFF);
}

int CSCDCCHeader::getCDFBunchCounter() const { 
  return ((word[0]>>20)&0xFFF);
}
int CSCDCCHeader::getCDFSourceId() const { 
  return ((word[0]>>8)&0xFFF);
}
int CSCDCCHeader::getCDFFOV() const { 
  return ((word[0]>>4)&0xF);
}
int CSCDCCHeader::getCDFEventType() const { 
  return ((word[0]>>56)&0xF);
}


std::ostream & operator<<(std::ostream & os, const CSCDCCHeader & hdr) {
  os << "DCC Header" << std::endl;
  return os;
}

