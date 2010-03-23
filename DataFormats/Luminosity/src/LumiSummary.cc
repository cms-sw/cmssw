
// $Id: LumiSummary.cc,v 1.14 2010/03/23 15:02:14 xiezhen Exp $

#include "DataFormats/Luminosity/interface/LumiSummary.h"

#include <iomanip>
#include <ostream>
#include <iostream>
float
LumiSummary::avgInsDelLumi()const{ 
  return avginsdellumi_;
}
float
LumiSummary::avgInsDelLumiErr()const{ 
  return  avginsdellumierr_;
}
short
LumiSummary::lumiSecQual()const {
  return lumisecqual_; 
}
unsigned long long
LumiSummary::deadcount() const{
  return deadcount_;
}
float 
LumiSummary::deadFrac() const { 
  if(numorbit_==0) return -99.9;
  return float(deadcount_/(numorbit_*3564.0));
}
float 
LumiSummary::liveFrac() const { 
  if(numorbit_==0) return -99.9;
  return (1.0f - float(deadcount_/(numorbit_*3564.0))); 
}
unsigned int 
LumiSummary::lsNumber() const{
  return lsnumber_; 
}
unsigned int
LumiSummary::startOrbit() const{
  return startorbit_; 
}
unsigned int
LumiSummary::numOrbit() const{
  return numorbit_;
}
bool 
LumiSummary::isValid() const {
  return (lumiversion_!="-1"); 
}
LumiSummary::L1   
LumiSummary::l1info(const std::string& name) const{
  for(std::vector<L1>::const_iterator it=l1data_.begin();it!=l1data_.end();++it){
    if(it->triggername==name) return *it;
  }
  return LumiSummary::L1();
}
LumiSummary::L1  
LumiSummary::l1info(unsigned int idx)const{
  return l1data_.at(idx);
}
LumiSummary::HLT  
LumiSummary::hltinfo(unsigned int idx) const {
  return hltdata_.at(idx);
}
LumiSummary::HLT  
LumiSummary::hltinfo(const std::string& pathname) const {
  for(std::vector<HLT>::const_iterator it=hltdata_.begin();it!=hltdata_.end();++it){
    if(it->pathname==pathname) return *it;
  }
  return LumiSummary::HLT();
}
size_t
LumiSummary::nTriggerLine()const{
  return l1data_.size();
}
size_t
LumiSummary::nHLTPath()const{
  return hltdata_.size();
}
std::vector<std::string>
LumiSummary::HLTPaths()const{
  std::vector<std::string> result;
  for(std::vector<HLT>::const_iterator it=hltdata_.begin();it!=hltdata_.end();++it){
    result.push_back(it->pathname);
  }
  return result;
}
float
LumiSummary::avgInsRecLumi() const {
  return avginsdellumi_ * liveFrac(); 
}
float
LumiSummary::avgInsRecLumiErr() const {
  return avginsdellumierr_ * liveFrac(); 
}
bool
LumiSummary::isProductEqual(LumiSummary const& next) const {
  return (avginsdellumi_ == next.avginsdellumi_ &&
          avginsdellumierr_ == next.avginsdellumierr_ &&
          lumisecqual_ == next.lumisecqual_ &&
          deadcount_ == next.deadcount_ &&
          lsnumber_ == next.lsnumber_ &&
	  startorbit_== next.startorbit_ &&
	  numorbit_==next.numorbit_&&
          l1data_.size() == next.l1data_.size() &&
          hltdata_.size() == next.hltdata_.size() &&
	  lumiversion_ == next.lumiversion_ );
}
std::string 
LumiSummary::lumiVersion()const{
  return lumiversion_;
}
void
LumiSummary::setLumiVersion(const std::string& lumiversion){
  lumiversion_=lumiversion;
}
void 
LumiSummary::setLumiData(float instlumi,float instlumierr,short lumiquality){
  avginsdellumi_=instlumi;
  avginsdellumierr_=instlumierr;
  lumisecqual_=lumiquality;
}
void 
LumiSummary::setDeadtime(unsigned long long deadcount){
  deadcount_=deadcount;
}
void 
LumiSummary::setlsnumber(unsigned int lsnumber){
  lsnumber_=lsnumber;
}
void 
LumiSummary::setOrbitData(unsigned int startorbit,unsigned int numorbit){
  startorbit_=startorbit;
  numorbit_=numorbit;
}
void 
LumiSummary::swapL1Data(std::vector<L1>& l1data){
  l1data_.swap(l1data);
}
void 
LumiSummary::swapHLTData(std::vector<HLT>& hltdata){
  hltdata_.swap(hltdata);
}
void 
LumiSummary::copyL1Data(const std::vector<L1>& l1data){
  l1data_.assign(l1data.begin(),l1data.end());
}
void 
LumiSummary::copyHLTData(const std::vector<HLT>& hltdata){
  hltdata_.assign(hltdata.begin(),hltdata.end());
}
std::ostream& operator<<(std::ostream& s, const LumiSummary& lumiSummary) {
  s << "\nDumping LumiSummary\n\n";
  if(!lumiSummary.isValid()){
    s << " === Invalid Lumi values === \n";
  }
  s << "  lumiVersion = " << lumiSummary.lumiVersion()  << "\n";
  s << "  avgInsDelLumi = " << lumiSummary.avgInsDelLumi() << "\n";
  s << "  avgInsDelLumiErr = " << lumiSummary.avgInsDelLumiErr() << "\n";
  s << "  lumiSecQual = " << lumiSummary.lumiSecQual() << "\n";
  s << "  deadCount = " << lumiSummary.deadcount() << "\n";
  s << "  deadFrac = " << lumiSummary.deadFrac() << "\n";
  s << "  liveFrac = " << lumiSummary.liveFrac() << "\n";
  s << "  lsNumber = " << lumiSummary.lsNumber() << "\n";
  s << "  startOrbit = " << lumiSummary.startOrbit() <<"\n";
  s << "  numOrbit = " << lumiSummary.numOrbit() <<"\n";
  s << "  avgInsRecLumi = " << lumiSummary.avgInsRecLumi() << "\n";
  s << "  avgInsRecLumiErr = "  << lumiSummary.avgInsRecLumiErr() << "\n\n";
  s << std::setw(15) << "l1name";
  s << std::setw(15) << "l1count";
  s << std::setw(15) << "l1prescale";
  s << "\n";
  size_t nTriggers=lumiSummary.nTriggerLine();
  size_t nHLTPath=lumiSummary.nHLTPath();
  for(unsigned int i = 0; i < nTriggers; ++i) {
    s << std::setw(15);
    s << lumiSummary.l1info(i).triggername;
    
    s << std::setw(15);
    s << lumiSummary.l1info(i).ratecount;
    
    s << std::setw(15);
    s << lumiSummary.l1info(i).prescale;
    s<<"\n";
  }
  s << std::setw(15) << "hltpath";
  s << std::setw(15) << "hltcount";
  s << std::setw(15) << "hltprescale";
  s << std::setw(15) << "hltinput";
  s << "\n";
  for(unsigned int i = 0; i < nHLTPath; ++i) {
    s << std::setw(15);
    s << lumiSummary.hltinfo(i).pathname;
    s << std::setw(15);
    s << lumiSummary.hltinfo(i).ratecount;
    s << std::setw(15);
    s << lumiSummary.hltinfo(i).prescale;
    s << std::setw(15);
    s << lumiSummary.hltinfo(i).inputcount;
    s << "\n";
  }
  return s << "\n";
}
