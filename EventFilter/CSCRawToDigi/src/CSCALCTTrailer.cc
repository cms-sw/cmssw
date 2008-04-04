/** documented in  flags
  http://www.phys.ufl.edu/~madorsky/alctv/alct2000_spec.PDF
*/

#include "EventFilter/CSCRawToDigi/interface/CSCALCTTrailer.h"

bool CSCALCTTrailer::debug=false;
short unsigned int CSCALCTTrailer::firmwareVersion=2006; 



CSCALCTTrailer::CSCALCTTrailer() { ///needed for packing
  firmwareVersion = 2006;
  trailer2006.e0dLine = 0xe0d; 
  trailer2006.reserved_4=0xd;
}

CSCALCTTrailer::CSCALCTTrailer(const unsigned short * buf){
  ///determine the version first
  if ((buf[0]==0xDE0D)&&((buf[1]&0xF000)==0xD000)) {
    firmwareVersion=2007;
  }
  else if ( (buf[2]&0xFFF)==0xE0D ) {
    firmwareVersion=2006;
  }
  else {
    edm::LogError("CSCALCTTrailer") <<"failed to construct: undetermined ALCT firmware version!!";
  }

  ///Now fill data 
  switch (firmwareVersion) {
  case 2006:
    memcpy(&trailer2006, buf, trailer2006.sizeInWords()*2);
    break;
  case 2007:
    memcpy(&trailer2007, buf, trailer2007.sizeInWords()*2);
    break;
  default:
    edm::LogError("CSCALCTTrailer")
      <<"coundn't construct: ALCT firmware version is bad/not defined!";
    break;
  }
}
