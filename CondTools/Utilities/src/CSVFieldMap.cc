#include "CondTools/Utilities/interface/CSVFieldMap.h"
#include "CondCore/DBCommon/interface/Exception.h"
void CSVFieldMap::push_back(const std::string& fieldName, const std::string& fieldType){
  m_fieldMap.push_back(std::make_pair(fieldName, fieldType));
}
std::string CSVFieldMap::fieldName( int idx ) const{
  return m_fieldMap[idx].first;
}
const std::type_info& CSVFieldMap::fieldType( int idx ) const{
  if(m_fieldMap[idx].second=="CHAR"){
    return typeid(std::string);
  }
  if(m_fieldMap[idx].second=="INT"){
    return typeid(int);
  }
  if(m_fieldMap[idx].second=="UINT"){
    return typeid(unsigned int);
  }
  if(m_fieldMap[idx].second=="FLOAT"){
    return typeid(float);
  }
  if(m_fieldMap[idx].second=="DOUBLE"){
    return typeid(double);
  }
  throw cond::Exception(std::string("unrecognised CSV type: ")+m_fieldMap[idx].second);
}
std::string CSVFieldMap::fieldTypeName( int idx ) const{
  return m_fieldMap[idx].second;
}
int CSVFieldMap::size() const{
  return m_fieldMap.size();
}
