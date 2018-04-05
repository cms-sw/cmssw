#include "CondFormats/Luminosity/interface/LumiSectionData.h"
#include <iostream>
lumi::LumiSectionData::LumiSectionData(): m_sectionid(0),m_versionid("-1"){
  m_bx.reserve(lumi::BXMAX*LUMIALGOMAX);
}
std::string
lumi::LumiSectionData::lumiVersion()const{
  return m_versionid; 
}
int
lumi::LumiSectionData::lumisectionID()const{
  return m_sectionid;
}
size_t
lumi::LumiSectionData::nBunchCrossing()const{
  return m_bx.size()/lumi::LUMIALGOMAX;
}
float
lumi::LumiSectionData::lumiAverage()const{
  return m_lumiavg;
}
float 
lumi::LumiSectionData::lumiError()const{
  return  m_lumierror;
}
float 
lumi::LumiSectionData::deadFraction()const{
  return m_deadfrac;
}
unsigned long long
lumi::LumiSectionData::startorbit()const{
  return m_startorbit;
}
int 
lumi::LumiSectionData::lumiquality()const{
  return m_quality;
}
void 
lumi::LumiSectionData::bunchCrossingInfo(  const lumi::LumiAlgoType lumialgotype,std::vector<lumi::BunchCrossingInfo>& result )const {
  result.clear();
  size_t offset=lumialgotype*lumi::BXMAX;
  std::copy(m_bx.begin()+offset,m_bx.begin()+offset+lumi::BXMAX,std::back_inserter(result));
}

const lumi::BunchCrossingInfo 
lumi::LumiSectionData::bunchCrossingInfo( const int BXIndex,
				  const LumiAlgoType lumialgotype )const{
  int realIdx=BXIndex-lumi::BXMIN+lumialgotype*lumi::BXMAX;
  return m_bx.at(realIdx);
}
lumi::BunchCrossingIterator 
lumi::LumiSectionData::bunchCrossingBegin( const LumiAlgoType lumialgotype )const{
  return m_bx.begin()+lumialgotype*BXMAX;
}
lumi::BunchCrossingIterator 
lumi::LumiSectionData::bunchCrossingEnd( const LumiAlgoType lumialgotype )const{
  return m_bx.end()-(lumi::BXMAX)*lumialgotype;
}
size_t
lumi::LumiSectionData::nHLTPath()const{
  return  m_hlt.size();
}
bool 
lumi::LumiSectionData::HLThasData()const{
  return !m_hlt.empty();
}
lumi::HLTIterator
lumi::LumiSectionData::hltBegin()const{
  return m_hlt.begin();
}
lumi::HLTIterator
lumi::LumiSectionData::hltEnd()const{
  return m_hlt.end();
}
bool
lumi::LumiSectionData::TriggerhasData()const{
  return !m_trigger.empty();
}
lumi::TriggerIterator
lumi::LumiSectionData::trgBegin()const{
  return m_trigger.begin();
}
lumi::TriggerIterator
lumi::LumiSectionData::trgEnd()const{
  return m_trigger.end();
}
short
lumi::LumiSectionData::qualityFlag()const{
  return m_quality;
}
void
lumi::LumiSectionData::setLumiNull(){
  m_versionid=-99;
}
void 
lumi::LumiSectionData::setLumiVersion(const std::string& versionid){
  m_versionid=versionid;
}
void
lumi::LumiSectionData::setLumiSectionId(int sectionid){
  m_sectionid=sectionid;
}
void
lumi::LumiSectionData::setLumiAverage(float avg){
  m_lumiavg=avg;
}
void 
lumi::LumiSectionData::setLumiQuality(int lumiquality){
  m_quality=lumiquality;
}
void 
lumi::LumiSectionData::setDeadFraction(float deadfrac){
  m_deadfrac=deadfrac;
}
void 
lumi::LumiSectionData::setLumiError(float lumierr){
  m_lumierror=lumierr;
}
void
lumi::LumiSectionData::setStartOrbit(unsigned long long orbtnumber){
  m_startorbit=orbtnumber;
}
void 
lumi::LumiSectionData::setBunchCrossingData(const std::vector<BunchCrossingInfo>& BXs,const LumiAlgoType algotype){
  std::copy(BXs.begin(),BXs.begin()+lumi::BXMAX,std::back_inserter(m_bx));
}
void 
lumi::LumiSectionData::setHLTData(const std::vector<HLTInfo>& hltdetail){
  std::copy(hltdetail.begin(),hltdetail.end(),std::back_inserter(m_hlt));
}
void
lumi::LumiSectionData::setTriggerData(const std::vector<TriggerInfo>& triggerinfo){
  std::copy(triggerinfo.begin(),triggerinfo.end(),std::back_inserter(m_trigger));
}
void 
lumi::LumiSectionData::setQualityFlag(short qualityflag){
  m_quality=qualityflag;
}
void
lumi::LumiSectionData::print( std::ostream& s ) const{
  s<<"lumi section id :"<<m_sectionid <<", ";
  s<<"lumi data version : "<<m_versionid<<", ";
  s<<"lumi average : "<<m_lumiavg<<", ";
  s<<"lumi error : "<<m_lumierror<<", ";
  s<<"lumi quality : "<<m_quality<<", ";
  s<<"lumi deadfrac : "<<m_deadfrac<<std::endl;
  std::vector<lumi::TriggerInfo>::const_iterator trgit;
  std::vector<lumi::TriggerInfo>::const_iterator trgitBeg=m_trigger.begin();
  std::vector<lumi::TriggerInfo>::const_iterator trgitEnd=m_trigger.end();
  unsigned int i=0;
  for(trgit=trgitBeg;trgit!=trgitEnd;++trgit){
    std::cout<<"  trg "<<i<<" : name : "<<trgit->name<<" : count : "<<trgit->triggercount<<" : deadtime : "<< trgit->deadtimecount<<" : prescale : "<<trgit->prescale<<std::endl;
    ++i;
  }
}
