#ifndef DQMOFFLINE_TRIGGER_ELETYPECODES
#define DQMOFFLINE_TRIGGER_ELETYPECODES

//defines bitwise commincation codes for the electron type

#include "DQMOffline/Trigger/interface/ComCodes.h"


#include <cstring>
#include <map>
#include <string>
#include <iostream>

class EleTypeCodes { 

public:
 
  enum EleTypeCode{
    //common cuts
    BARREL     = 0x0001,
    ENDCAP     = 0x0002,
    GOLDEN     = 0x0004,
    NARROW     = 0x0008,
    BIGBREM    = 0x0010,
    SHOWERING  = 0x0020,
    CRACK      = 0x0040
  };

private:
  static ComCodes codes_;

private:

  EleTypeCodes(){} //no instanitation possible
  ~EleTypeCodes(){}

public:

  static int makeTypeCode(int eleType);
  static int getCode(const char *descript){return codes_.getCode(descript);}
  void getCodeName(int code,std::string& id){return codes_.getCodeName(code,id);}

private:
  static ComCodes setCodes_();

};

#endif
