#ifndef DQMOFFLINE_TRIGGER_CUTCODES
#define DQMOFFLINE_TRIGGER_CUTCODES


#include "DQMOffline/Trigger/interface/ComCodes.h"

#include <cstring>
#include <map>
#include <string>
#include <iostream>

class CutCodes { //class to handle the cutcodes used in electron cutting

public:
  //redefining the codes to be unique (currently use the same ones for some cem only and plug only cuts)
  //this will save a lot of hassle (cem uses first word, pem uses 2nd word
  //with common cuts using the first nibble
  //above comment very out of date (note references to CDF detector...)
  

  enum CutCode{
    //common cuts
   ET            =0x0001,
   PT            =0x0002,
   DETETA        =0x0004,
   CRACK         =0x0008,

   EPIN          =0x0010,
   DETAIN        =0x0020,
   DPHIIN        =0x0040,
   HADEM         =0x0080,
   EPOUT         =0x0100,
   DPHIOUT       =0x0200,
   INVEINVP      =0x0400,
   BREMFRAC      =0x0800,
   E9OVERE25     =0x1000,
   SIGMAETAETA   =0x2000,
   SIGMAPHIPHI   =0x4000,
   ISOLEM        =0x8000,
   ISOLHAD       =0x00010000,
   ISOLPTTRKS    =0x00020000,
   ISOLNRTRKS    =0x00040000,

   //flag that if its set, shows the code is invalid
   INVALID       =0x10000000
 
  };

private:
  static ComCodes codes_;

private:
  CutCodes(){} //not going to allow instainitiation
  ~CutCodes(){}

public:
  static int getCode(const char *descript){return codes_.getCode(descript);}
  static void getCodeName(int code,std::string& id){return codes_.getCodeName(code,id);}

private:
  static ComCodes setCodes_();
  
};

#endif
