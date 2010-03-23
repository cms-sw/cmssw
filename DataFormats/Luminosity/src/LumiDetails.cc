
// $Id: LumiDetails.cc,v 1.8 2010/03/23 15:56:03 xiezhen Exp $

#include "DataFormats/Luminosity/interface/LumiDetails.h"

#include <iomanip>

//using namespace std;
static std::vector<float> emptyFloatVec;
static std::vector<short> emptyShortVec;
LumiDetails::LumiDetails():m_lumiversion("-1"){}
LumiDetails::LumiDetails(const std::string& lumiversion):m_lumiversion(lumiversion){
}
LumiDetails::~LumiDetails(){
  m_lumivalueMap.clear();
  m_lumierrorMap.clear();
  m_lumiqualityMap.clear();
}
void
LumiDetails::setLumiVersion(const std::string& lumiversion){
  m_lumiversion=lumiversion;
}
std::string 
LumiDetails::lumiVersion()const{
  return m_lumiversion;
}
bool
LumiDetails::isValid() const{
  return (m_lumiversion!="-1"); 
}
void 
LumiDetails::swapValueData(std::map<std::string,std::vector<float> >& data){
  data.swap(m_lumivalueMap);
}
void 
LumiDetails::swapErrorData(std::map<std::string,std::vector<float> >& data){
  data.swap(m_lumierrorMap);
}
void 
LumiDetails::swapQualData(std::map<std::string,std::vector<short> >& data){
  data.swap(m_lumiqualityMap);
}
void 
LumiDetails::copyValueData(const std::map<std::string,std::vector<float> >& data){
  m_lumivalueMap=data;
}
void 
LumiDetails::copyErrorData(const std::map<std::string,std::vector<float> >& data){
  m_lumierrorMap=data;
}
void 
LumiDetails::copyQualData(const std::map<std::string,std::vector<short> >& data){
  m_lumiqualityMap=data;
}
float 
LumiDetails::lumiValue(const std::string& algoname,unsigned int bx) const{
  if(bx>0){
    std::map< std::string,std::vector<float> >::const_iterator it=m_lumivalueMap.find(algoname);
    if(it!=m_lumivalueMap.end()){
      return it->second.at(bx);
    }
  }
  return -1.0;
}
float 
LumiDetails::lumiError(const std::string& algoname,unsigned int bx) const{
  if(bx>0){
    std::map< std::string,std::vector<float> >::const_iterator it=m_lumierrorMap.find(algoname);
    if(it!=m_lumierrorMap.end()){
      return it->second.at(bx);
    }
  }
  return -1.0;
}
short 
LumiDetails::lumiQuality(const std::string& algoname,unsigned int bx) const{
  if(bx>0){
    std::map< std::string,std::vector<short> >::const_iterator it=m_lumiqualityMap.find(algoname);
    if(it!=m_lumiqualityMap.end()){
      return it->second.at(bx);
    }
  }
  return -1.0;
}
  
const std::vector<float>& 
LumiDetails::lumiValuesForAlgo(const std::string& algoname) const{
  std::map< std::string,std::vector<float> >::const_iterator it;
  it=m_lumivalueMap.find(algoname);
  if(it!=m_lumivalueMap.end()){
    return it->second;
  }else{
    return emptyFloatVec;
  }
}
const std::vector<float>& 
LumiDetails::lumiErrorsForAlgo(const std::string& algoname) const{
  std::map< std::string,std::vector<float> >::const_iterator it;
  it=m_lumierrorMap.find(algoname);
  if(it!=m_lumierrorMap.end()){
    return it->second;
  }else{
    return emptyFloatVec;
  }
}
const std::vector<short>& 
LumiDetails::lumiQualsForAlgo(const std::string& algoname) const{
  std::map< std::string,std::vector<short> >::const_iterator it;
  it=m_lumiqualityMap.find(algoname);
  if(it!=m_lumiqualityMap.end()){
    return it->second;
  }else{
    return emptyShortVec;
  }
}
const std::map< std::string,std::vector<float> >&
LumiDetails::allLumiValues()const{
  return  m_lumivalueMap;
}
const std::map< std::string,std::vector<float> >& 
LumiDetails::allLumiErrors()const{
  return m_lumierrorMap; 
}
const std::map< std::string,std::vector<short> >& 
LumiDetails::allLumiQuals()const{
  return m_lumiqualityMap;
}
std::vector<std::string>
LumiDetails::algoNames()const{
  std::vector<std::string> result;
  std::map< std::string,std::vector<float> >::const_iterator it;
  std::map< std::string,std::vector<float> >::const_iterator itBeg=m_lumivalueMap.begin();
  std::map< std::string,std::vector<float> >::const_iterator itEnd=m_lumivalueMap.end();
  for(it=itBeg;it!=itEnd;++it){
    result.push_back(it->first);
  }
  return result;
}
unsigned int 
LumiDetails::totalLumiAlgos()const{
  return m_lumivalueMap.size();
}

//bool
//LumiDetails::isProductEqual(LumiDetails const& next) const {
//}

std::ostream& operator<<(std::ostream& s, const LumiDetails& lumiDetails) {
  
  s << "\nDumping LumiDetails\n";
  s << std::setw(12) << "lumi version" <<lumiDetails.lumiVersion() << "\n";

  std::vector<std::string> algoNames=lumiDetails.algoNames();
  std::vector<std::string>::const_iterator it;
  std::vector<std::string>::const_iterator itBeg=algoNames.begin();
  std::vector<std::string>::const_iterator itEnd=algoNames.end();

  std::vector<float>::const_iterator valueit;
  std::vector<float>::const_iterator errorit;
  std::vector<short>::const_iterator qualityit;
  for(it=itBeg;it!=itEnd;++it){
    const std::vector<float>& lumivalues=lumiDetails.lumiValuesForAlgo(*it);
    
    const std::vector<float>& lumierrors=lumiDetails.lumiErrorsForAlgo(*it);
    errorit=lumierrors.begin();
    const std::vector<short>& lumiqualities=lumiDetails.lumiQualsForAlgo(*it);
    qualityit=lumiqualities.begin();
    s << std::setw(12);
    s << "algorithm:"<<(*it)<< "\n";
    s << std::setw(12)<< "value"<<std::setw(12)<< "error"<<std::setw(12)<< "quality"<<"\n";
    for(valueit=lumivalues.begin(); valueit!=lumivalues.end(); ++valueit,++errorit,++qualityit){
      s <<std::setw(12)<<(*valueit)<<std::setw(12)<<(*errorit)<<std::setw(12)<<(*qualityit)<<"\n";
    }
    s<<"\n";
  }
  return s<<"\n";
}
