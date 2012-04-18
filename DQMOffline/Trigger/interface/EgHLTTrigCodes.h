#ifndef DQMOFFLINE_TRIGGER_EGHLTTRIGCODES
#define DQMOFFLINE_TRIGGER_EGHLTTRIGCODES


//author: Sam Harper
//aim: to define the trigger bits we are interested in
//implimentation: likely to be more than 32 (or even 64 bits) so differs from CutCodes in the fact it stores the bit position, not the bit mask

#include <cstring>
#include <string>
#include <iostream>
#include <bitset>
#include <vector>
#include <algorithm>

//unforunately hard coded limit of 64 bits which needs to be checked when setting it up
//if this becomes a problem, will be replaced by boost::dynamic_bitset
//my appologies for the typedef, it was better this way 

namespace egHLT {
  class TrigCodes {
    
  public:
    static const int maxNrBits_=128;
    typedef std::bitset<maxNrBits_> TrigBitSet;
    
    class TrigBitSetMap {
      
    private:
      //sorted vector
      std::vector<std::pair<std::string,TrigBitSet> > codeDefs_;
      
    public:
      TrigBitSetMap(){}
      ~TrigBitSetMap(){}
      
    public:
      TrigBitSet getCode(const char *descript)const;
      void getCodeName(TrigBitSet code,std::string& id)const;
      
      //modifiers
      void setCode(const char *descript,TrigBitSet code);
      void setCode(const char *descript,int bitNr);
      
      //key comp
      static bool keyComp(const std::pair<std::string,TrigBitSet>& lhs,const std::pair<std::string,TrigBitSet>& rhs);
      void sort(){std::sort(codeDefs_.begin(),codeDefs_.end(),keyComp);}
      size_t size()const{return codeDefs_.size();}
      void printCodes();
    };
    
    
  private:
    static TrigBitSetMap trigBitSetMap_;
    
  private:
    TrigCodes(){}//not allowed to instanstiate
    ~TrigCodes(){}
    
  public:
    // static void setCode(const char *descript,TrigBitSet code){trigBitSetMap_.setCode(descript,code);}
  //static void setCode(const char *descript,int bitNr){trigBitSetMap_.setCode(descript,bitNr);}
    static void setCodes(std::vector<std::string>& filterNames);
    
    
    static TrigBitSet getCode(const std::string& descript){return trigBitSetMap_.getCode(descript.c_str());}
    static void getCodeName(TrigBitSet code,std::string& id){return trigBitSetMap_.getCodeName(code,id);}
    static int maxNrBits(){return maxNrBits_;}
    static void printCodes(){return trigBitSetMap_.printCodes();}
    
  };
}

#endif
