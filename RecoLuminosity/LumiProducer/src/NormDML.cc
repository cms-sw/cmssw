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
std::vector< std::pair< unsigned int,lumi::NormDML::normData > >::const_iterator 
lumi::NormDML::normById(unsigned long long normid)const{
  return m_data.begin();
}
