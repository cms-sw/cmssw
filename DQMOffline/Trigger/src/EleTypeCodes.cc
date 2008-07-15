#include "DQMOffline/Trigger/interface/EleTypeCodes.h"

ComCodes EleTypeCodes::codes_ = EleTypeCodes::setCodes_();

ComCodes EleTypeCodes::setCodes_()
{
  ComCodes codes;
  codes.setCode("barrel",BARREL);
  codes.setCode("endcap",ENDCAP);
  codes.setCode("golden",GOLDEN);
  codes.setCode("narrow",NARROW);
  codes.setCode("bigBrem",BIGBREM);
  codes.setCode("showering",SHOWERING);
  codes.setCode("crack",CRACK);
  return codes;
}

int EleTypeCodes::makeTypeCode(int eleType)
{
  int typeCode=0x0;
  if(eleType<100) typeCode |=BARREL;
  else{
    typeCode |=ENDCAP;
    eleType-=100;
  }
  if(eleType/10==0) typeCode |=GOLDEN;
  else if(eleType/10==1) typeCode |= NARROW;
  else if(eleType/10==2) typeCode |= BIGBREM;
  else if(eleType/10==3) typeCode |= SHOWERING;
  else if(eleType/10==4) typeCode |= CRACK;
  else std::cout <<"EleTypeCode::makeTypeCode Warning :ele type "<<eleType<<" unrecognised "<<std::endl;
  return typeCode;
}
