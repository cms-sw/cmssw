#ifndef DQMOFFLINE_TRIGGER_EGHLTCOMCODES
#define DQMOFFLINE_TRIGGER_EGHLTCOMCODES

#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

namespace egHLT {

  class ComCodes { 
    
  private:
    std::vector<std::pair<std::string,int> > _codeDefs;
    
  public:
    ComCodes()= default; 
    ComCodes(const ComCodes& rhs)= default;
    ~ComCodes()= default; 
    
    //accessors
    int getCode(const char *descript)const;
    void getCodeName(int code,std::string& id)const;
    
    //modifiers
    void setCode(const char *descript,int code);
    
    //key comp
    static bool keyComp(const std::pair<std::string,int>& lhs,const std::pair<std::string,int>& rhs);
    void sort(){std::sort(_codeDefs.begin(),_codeDefs.end(),keyComp);}
    
  };
}
#endif
  
