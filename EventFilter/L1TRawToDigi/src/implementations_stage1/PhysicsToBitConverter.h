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
#include "rctDataBase.h"

namespace l1t{
  class PhysicsToBitConverter {


    int words32bitLink[2][6];  //[link][word]

    std::vector<int> bitsLink[2];
    rctDataBase databaseobject;


  public:

    PhysicsToBitConverter();
    ~PhysicsToBitConverter() { }
    void Set32bitWordLinkEven(int index,uint32_t value){words32bitLink[0][index]=value;};
    void Set32bitWordLinkOdd(int index,uint32_t value){words32bitLink[1][index]=value;};
    void Convert();

    int GetObject(rctDataBase::rctObjectType t, int firstindex, int secondindex = -1);

    int ReadBitInInt(int bit,int value);
    int BuildPhysicsValue(int firstbit,int bitlength,int linkid);
    
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


  };
}
#endif
