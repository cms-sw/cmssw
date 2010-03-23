
// $Id: LumiDetails.cc,v 1.7 2010/03/23 15:02:14 xiezhen Exp $

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
/**
bool
LumiDetails::isProductEqual(LumiDetails const& next) const {
  return (lumietsum_ == next.lumietsum_ &&
          lumietsumerr_ == next.lumietsumerr_ &&
          lumietsumqual_ == next.lumietsumqual_ &&
          lumiocc_ == next.lumiocc_ &&
          lumioccerr_ == next.lumioccerr_ &&
          lumioccqual_ == next.lumioccqual_);
}
**/

std::ostream& operator<<(std::ostream& s, const LumiDetails& lumiDetails) {
  /**
  const std::vector<float>& lumietsum     = lumiDetails.lumiEtSum();
  const std::vector<float>& lumietsumerr  = lumiDetails.lumiEtSumErr();
  const std::vector<int>& lumietsumqual = lumiDetails.lumiEtSumQual();
  const std::vector<float>& lumiocc       = lumiDetails.lumiOcc();
  const std::vector<float>& lumioccerr    = lumiDetails.lumiOccErr();
  const std::vector<int>& lumioccqual   = lumiDetails.lumiOccQual();

  unsigned int maxSize = lumietsum.size();
  if (lumietsumerr.size() > maxSize) maxSize = lumietsumerr.size();
  if (lumietsumqual.size() > maxSize) maxSize = lumietsumqual.size();
  if (lumiocc.size() > maxSize) maxSize = lumiocc.size();
  if (lumioccerr.size() > maxSize) maxSize = lumioccerr.size();
  if (lumioccqual.size() > maxSize) maxSize = lumioccqual.size();

  s << "\nDumping LumiDetails\n";
  s << setw(12) << "etsum";
  s << setw(12) << "etsumerr";
  s << setw(12) << "etsumqual";
  s << setw(12) << "occ";
  s << setw(12) << "occerr";
  s << setw(12) << "occqual";
  s << "\n";

  for (unsigned int i = 0; i < maxSize; ++i) {

    s << setw(12);
    i < lumietsum.size() ? s << lumietsum[i] : s << " ";

    s << setw(12);
    i < lumietsumerr.size() ? s << lumietsumerr[i] : s << " ";

    s << setw(12);
    i < lumietsumqual.size() ? s << lumietsumqual[i] : s << " ";

    s << setw(12);
    i < lumiocc.size() ? s << lumiocc[i] : s << " ";

    s << setw(12);
    i < lumioccerr.size() ? s << lumioccerr[i] : s << " ";

    s << setw(12);
    i < lumioccqual.size() ? s << lumioccqual[i] : s << " ";

    s << "\n";
    }
  **/
  return s << "\n";
}
