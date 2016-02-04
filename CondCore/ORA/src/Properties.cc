#include "CondCore/ORA/interface/Properties.h"

ora::Properties::Properties():m_properties(),m_flags(){
}

ora::Properties::~Properties(){
}

bool ora::Properties::setProperty(const std::string& propertyName, const std::string& propertyValue){
  bool ret = false;
  if(m_properties.find(propertyName)!=m_properties.end()){
    ret = true;
    m_properties.erase(propertyName);
  }
  m_properties.insert(std::make_pair(propertyName,propertyValue));
  return ret;
}

void ora::Properties::setFlag(const std::string& flagName){
  m_flags.insert(flagName);
}

bool ora::Properties::hasProperty(const std::string& propertyName) const {
  return (m_properties.find(propertyName)!=m_properties.end());
}

std::string ora::Properties::getProperty(const std::string& propertyName) const {
  std::string ret("");
  std::map<std::string,std::string>::const_iterator iP = m_properties.find(propertyName);
  if(iP!=m_properties.end()){
    ret = iP->second;
  }
  return ret;
}

bool ora::Properties::getFlag(const std::string& flagName) const {
  bool ret = false;
  if(m_flags.find(flagName)!=m_flags.end()){
    ret = true;
  }
  return ret;
}

bool ora::Properties::removeProperty(const std::string& propertyName){
  bool ret = false;
  if(m_properties.find(propertyName)!=m_properties.end()){
    ret = true;
    m_properties.erase(propertyName);
  }
  return ret;
}

bool ora::Properties::removeFlag(const std::string& flagName){
  bool ret = false;
  if(m_flags.find(flagName)!=m_flags.end()){
    ret = true;
    m_flags.erase(flagName);
  }
  return ret;
}


