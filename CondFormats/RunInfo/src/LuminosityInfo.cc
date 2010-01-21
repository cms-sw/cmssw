#include "CondFormats/RunInfo/interface/LuminosityInfo.h"
//#include <iostream>
lumi::LuminosityInfo::LuminosityInfo(): m_sectionid(0),m_versionid(-99){
  m_bx.reserve(lumi::BXMAX*LUMIALGOMAX);
}
bool 
lumi::LuminosityInfo::isNullData() const{
  return m_versionid<0;
}
short
lumi::LuminosityInfo::lumiVersionNumber()const{
  return m_sectionid; 
}
int
lumi::LuminosityInfo::lumisectionID()const{
  return m_sectionid;
}
size_t
lumi::LuminosityInfo::nBunchCrossing()const{
  return m_bx.size()/lumi::LUMIALGOMAX;
}
float
lumi::LuminosityInfo::lumiAverage()const{
  return m_lumiavg;
}
float 
lumi::LuminosityInfo::lumiError()const{
  return  m_lumierror;
}
float 
lumi::LuminosityInfo::deadFraction()const{
  return m_deadfrac;
}
int 
lumi::LuminosityInfo::lumiquality()const{
  return m_quality;
}
void 
lumi::LuminosityInfo::bunchCrossingInfo(  const lumi::LumiAlgoType lumialgotype, 
		    std::vector<lumi::BunchCrossingInfo>& result )const {
  result.clear();
  size_t offset=lumialgotype*lumi::BXMAX;
  std::copy(m_bx.begin()+offset,m_bx.begin()+offset+lumi::BXMAX,std::back_inserter(result));
}

const lumi::BunchCrossingInfo 
lumi::LuminosityInfo::bunchCrossingInfo( const int BXIndex,
				  const LumiAlgoType lumialgotype )const{
  int realIdx=BXIndex-lumi::BXMIN+lumialgotype*lumi::BXMAX;
  return m_bx.at(realIdx);
}
lumi::BunchCrossingIterator 
lumi::LuminosityInfo::bunchCrossingBegin( const LumiAlgoType lumialgotype )const{
  return m_bx.begin()+lumialgotype*BXMAX;
}
lumi::BunchCrossingIterator 
lumi::LuminosityInfo::bunchCrossingEnd( const LumiAlgoType lumialgotype )const{
  return m_bx.end()-(lumi::BXMAX)*lumialgotype;
}
void
lumi::LuminosityInfo::setLumiNull(){
  m_versionid=-99;
}
void 
lumi::LuminosityInfo::setLumiVersionNumber(short versionid){
  m_versionid=versionid;
}
void
lumi::LuminosityInfo::setLumiSectionId(int sectionid){
  m_sectionid=sectionid;
}
void
lumi::LuminosityInfo::setLumiAverage(float avg){
  m_lumiavg=avg;
}
void 
lumi::LuminosityInfo::setLumiQuality(int lumiquality){
  m_quality=lumiquality;
}
void 
lumi::LuminosityInfo::setDeadFraction(float deadfrac){
  m_deadfrac=deadfrac;
}
void 
lumi::LuminosityInfo::setLumiError(float lumierr){
  m_lumierror=lumierr;
}
void 
lumi::LuminosityInfo::setBunchCrossingData(const std::vector<BunchCrossingInfo>& BXs,const LumiAlgoType algotype){
  std::copy(BXs.begin(),BXs.begin()+lumi::BXMAX,std::back_inserter(m_bx));
}

