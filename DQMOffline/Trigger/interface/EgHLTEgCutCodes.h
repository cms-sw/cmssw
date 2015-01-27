#ifndef DQMOFFLINE_TRIGGER_EGHLTEGCUTCODES
#define DQMOFFLINE_TRIGGER_EGHLTEGCUTCODES


#include "DQMOffline/Trigger/interface/EgHLTComCodes.h"

#include <cstring>

#include <string>
#include <iostream>

namespace egHLT {
  class EgCutCodes { //class to handle the cutcodes used in electron cutting
    
  public:
    
    enum CutCode{
      //kinematic and fiducial cuts
      ET            =0x00000001,
      PT            =0x00000002,
      DETETA        =0x00000004,
      CRACK         =0x00000008,
      //track cuts
      DETAIN        =0x00000010,
      DPHIIN        =0x00000020,
      INVEINVP      =0x00000040,
    
      //supercluster cuts
      SIGMAETAETA   =0x00000080,
      HADEM         =0x00000100,
      SIGMAIETAIETA =0x00000200,
      E2X5OVER5X5   =0x00000400,
      //---Morse------
      //R9            =0x00000800,
      MINR9         =0x00000800,
      MAXR9         =0x00100000,
      //--------------
      //std isolation cuts
      ISOLEM        =0x00001000,
      ISOLHAD       =0x00002000,
      ISOLPTTRKS    =0x00004000,
      ISOLNRTRKS    =0x00008000,
      //hlt isolation cuts
      HLTISOLTRKSELE=0x00010000,
      HLTISOLTRKSPHO=0x00020000,
      HLTISOLHAD    =0x00040000,
      HLTISOLEM     =0x00080000,
      //track quaility cuts (hlt track algo isnt very forgiving)
      CTFTRACK      =0x00010000,
      //hlt quantities that are slightly different to reco
      HLTDETAIN     =0x00020000,
      HLTDPHIIN     =0x00040000,
      HLTINVEINVP   =0x00080000,
      //flag that if its set, shows the code is invalid
      INVALID       =0x80000000
      
    };

  private:
    static const ComCodes codes_;
    
  private:
    EgCutCodes(){} //not going to allow instainitiation
    ~EgCutCodes(){}
    
  public:
    static int getCode(const std::string& descript){return codes_.getCode(descript.c_str());}
    static void getCodeName(int code,std::string& id){return codes_.getCodeName(code,id);}
    
  private:
    static ComCodes setCodes_();
    
  };
}
#endif
