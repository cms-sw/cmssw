#ifndef _PHYSICSTOBITCONVERTER_h
#define _PHYSICSTOBITCONVERTER_h

#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <stdint.h>
#include <iomanip>
#include <sstream>
#include <vector>
#include <array>
#include <bitset>
#include <stdint.h>
#include <math.h>
#include "rctDataBase.h"

namespace l1t{
  class PhysicsToBitConverter {


    int words32bitLink[2][6];  //[link][word]

    int bitsLink[2][192];
    rctDataBase databaseobject;


  public:

    PhysicsToBitConverter();
    ~PhysicsToBitConverter() { }
    void Set32bitWordLinkEven(int index,uint32_t value){words32bitLink[0][index]=value;};
    void Set32bitWordLinkOdd(int index,uint32_t value){words32bitLink[1][index]=value;};
    
    int Get32bitWordLinkEven(int index){return words32bitLink[0][index];};
    int Get32bitWordLinkOdd(int index) {return words32bitLink[1][index];};
    
    void Convert();
    void Extract32bitwords();

    int GetObject(rctDataBase::rctObjectType t, int firstindex, int secondindex = -1);
    void SetObject(rctDataBase::rctObjectType t, int value, int firstindex, int secondindex = -1);

    int ReadBitInInt(int bit,int value);
    int BuildDecimalValue(int firstbit,int bitlength,int linkid);
    
    int GetRCEt(int card,int region) {return GetObject(rctDataBase::RCEt,card,region);}
    int GetHFEt(int region)          {return GetObject(rctDataBase::HFEt,region);}
    int GetRCTau(int card,int region){return GetObject(rctDataBase::RCTau,card,region);}
    int GetRCOf(int card,int region) {return GetObject(rctDataBase::RCOf,card,region);}
    int GetHFFg(int region)          {return GetObject(rctDataBase::HFFg,region);}
    int GetNEReg(int cand)           {return GetObject(rctDataBase::NEReg,cand);}
    int GetNECard(int cand)           {return GetObject(rctDataBase::NECard,cand);}
    int GetNEEt(int cand)            {return GetObject(rctDataBase::NEEt,cand);}
    int GetIEReg(int cand)           {return GetObject(rctDataBase::IEReg,cand);}
    int GetIECard(int cand)           {return GetObject(rctDataBase::IECard,cand);}
    int GetIEEt(int cand)            {return GetObject(rctDataBase::IEEt,cand);}
    int GetRCHad(int card,int region){return GetObject(rctDataBase::RCHad,card,region);}
    
    void SetRCEt(int value,int card,int region) {SetObject(rctDataBase::RCEt,value,card,region);}
    void SetHFEt(int value,int region)          {SetObject(rctDataBase::HFEt,value,region);}
    void SetRCTau(int value,int card,int region){SetObject(rctDataBase::RCTau,value,card,region);}
    void SetRCOf(int value,int card,int region) {SetObject(rctDataBase::RCOf,value,card,region);}
    void SetHFFg(int value,int region)          {SetObject(rctDataBase::HFFg,value,region);}
    void SetNEReg(int value,int cand)           {SetObject(rctDataBase::NEReg,value,cand);}
    void SetNECard(int value,int cand)          {SetObject(rctDataBase::NECard,value,cand);}
    void SetNEEt(int value,int cand)            {SetObject(rctDataBase::NEEt,value,cand);}
    void SetIEReg(int value,int cand)           {SetObject(rctDataBase::IEReg,value,cand);}
    void SetIECard(int value,int cand)          {SetObject(rctDataBase::IECard,value,cand);}
    void SetIEEt(int value,int cand)            {SetObject(rctDataBase::IEEt,value,cand);}
    void SetRCHad(int value,int card,int region){SetObject(rctDataBase::RCHad,value,card,region);}

  };
}
#endif
