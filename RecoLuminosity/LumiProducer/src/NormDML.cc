#include "RecoLuminosity/LumiProducer/interface/NormDML.h"
lumi::NormDML::NormDML(){
}
unsigned long long 
lumi::NormDML::normIdByName(const coral::ISchema& schema,const std::string& normtagname){
  return 0;
}
unsigned long long 
lumi::NormDML::normIdByType(const coral::ISchema& schema,LumiType,bool defaultonly){
  return 0;
}
void
lumi::NormDML::normById(const coral::ISchema&schema, unsigned long long normid, std::map< unsigned int,lumi::NormDML::normData >& result)const{
}
