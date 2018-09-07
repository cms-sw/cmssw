/** documented in  flags
  http://www.phys.ufl.edu/~madorsky/alctv/alct2000_spec.PDF
*/

#include "EventFilter/CSCRawToDigi/interface/CSCALCTTrailer.h"


#ifdef LOCAL_UNPACK

bool CSCALCTTrailer::debug=false;
short unsigned int CSCALCTTrailer::firmwareVersion=2006; 

#else

std::atomic<bool> CSCALCTTrailer::debug{false};
std::atomic<short unsigned int> CSCALCTTrailer::firmwareVersion{2006}; 

#endif

CSCALCTTrailer2006::CSCALCTTrailer2006() {
  bzero(this,  sizeInWords()*2); ///size of the trailer
  e0dLine = 0xDE0D;
  d_0=0xD;
  d_1=0xD;
  zero_0 = 0;
  zero_1 = 0;
  d_3 = 0xD;
  reserved_3 = 1;
}


CSCALCTTrailer2007::CSCALCTTrailer2007() {
  bzero(this,  sizeInWords()*2); ///size of the trailer
  e0dLine = 0xDE0D;
  reserved_0 = 0xD;
  reserved_1 = 0xD;
  reserved_3 = 1;
  reserved_4 = 0xD;
}



CSCALCTTrailer::CSCALCTTrailer(int size, int firmVersion) 
{ ///needed for packing
  if(firmVersion == 2006)
  {
     trailer2006.setSize(size);
     firmwareVersion = 2006;
  }
  else if (firmVersion == 2007)
  {
     trailer2007.setSize(size);
     firmwareVersion = 2007;
  }
  else {
    edm::LogError("CSCALCTTrailer|CSCRawToDigi") <<"failed to construct: undetermined ALCT firmware version!!" << firmVersion;
  }

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
    edm::LogError("CSCALCTTrailer|CSCRawToDigi") <<"failed to construct: undetermined ALCT firmware version!!" << firmwareVersion;
  }

  ///Now fill data 
#ifdef LOCAL_UNPACK
  switch (firmwareVersion) {
#else
  switch (firmwareVersion.load()) {
#endif
  case 2006:
    trailer2006.setFromBuffer(buf);
    break;
  case 2007:
    trailer2007.setFromBuffer(buf);
    break;
  default:
    edm::LogError("CSCALCTTrailer|CSCRawToDigi")
      <<"couldn't construct: ALCT firmware version is bad/not defined!";
    break;
  }
}
