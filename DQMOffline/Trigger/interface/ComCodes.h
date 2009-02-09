#ifndef DQMOFFLINE_TRIGGER_COMCODES
#define DQMOFFLINE_TRIGGER_COMCODES

#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

//used to be ComCodesBase, however changed the design that instead of inheriting from this class, all my communication classes
//would own an object of this class
//this has the added advantage that they can all be declered static and I wasnt using any of the inheritance features anyway
//this class is basically a map which stores the communication bit codes and their names so they can be accessed easily
class ComCodes { 

private:
  //std::map<std::string,int> _codeDefs;
  std::vector<std::pair<std::string,int> > _codeDefs;

public:
  ComCodes(){} 
  ComCodes(const ComCodes& rhs):_codeDefs(rhs._codeDefs){}
  ~ComCodes(){} 
  
  //accessors
  int getCode(const char *descript)const;
  void getCodeName(int code,std::string& id)const;
  
  //modifiers
  void setCode(const char *descript,int code);

  //key comp
  static bool keyComp(const std::pair<std::string,int>& lhs,const std::pair<std::string,int>& rhs);
  void sort(){std::sort(_codeDefs.begin(),_codeDefs.end(),keyComp);}

};

#endif
