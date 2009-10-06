
// $Id: LumiSummary.cc,v 1.6 2009/05/12 19:40:35 xiezhen Exp $

#include "DataFormats/Luminosity/interface/LumiSummary.h"

#include <iomanip>
#include <ostream>

using namespace std;

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
float 
LumiSummary::deadFrac() const { 
  return deadfrac_; 
}
float 
LumiSummary::liveFrac() const { 
  return (1.0f - deadfrac_); 
}
int 
LumiSummary::lsNumber() const{
  return lsnumber_; 
}
unsigned long long
LumiSummary::startOrbit() const{
  return startorbit_; 
}
bool 
LumiSummary::isValid() const {
  return (lsnumber_ > 0); 
}
LumiSummary::L1   
LumiSummary::l1info(int linenumber) const{
  return l1data_.at(linenumber);
}
std::string
LumiSummary::triggerConfig(int linenumber)const{
  return l1data_.at(linenumber).triggersource;
}
LumiSummary::HLT  
LumiSummary::hltinfo(int idx) const {
  return hltdata_.at(idx);
}
LumiSummary::HLT  
LumiSummary::hltinfo(const std::string& pathname) const {
  for(std::vector<HLT>::const_iterator it=hltdata_.begin();it!=hltdata_.end();++it){
    if(it->pathname==pathname) return *it;
  }
  return LumiSummary::HLT() ;
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
          deadfrac_ == next.deadfrac_ &&
          lsnumber_ == next.lsnumber_ &&
          l1data_.size() == next.l1data_.size() &&
          hltdata_.size() == next.hltdata_.size() );
}

std::ostream& operator<<(std::ostream& s, const LumiSummary& lumiSummary) {
  s << "\nDumping LumiSummary\n\n";
  s << "  avgInsDelLumi = " << lumiSummary.avgInsDelLumi() << "\n";
  s << "  avgInsDelLumiErr = " << lumiSummary.avgInsDelLumiErr() << "\n";
  s << "  lumiSecQual = " << lumiSummary.lumiSecQual() << "\n";
  s << "  deadFrac = " << lumiSummary.deadFrac() << "\n";
  s << "  liveFrac = " << lumiSummary.liveFrac() << "\n";
  s << "  lsNumber = " << lumiSummary.lsNumber() << "\n";
  s << "  startOrbit = " << lumiSummary.startOrbit() <<"\n";
  s << "  avgInsRecLumi = " << lumiSummary.avgInsRecLumi() << "\n";
  s << "  avgInsRecLumiErr = " << lumiSummary.avgInsRecLumiErr() << "\n\n";
  s << setw(15) << "l1source";
  s << setw(15) << "l1ratecounter";
  s << setw(15) << "l1scaler";
  s << "\n";
  size_t nTriggers=lumiSummary.nTriggerLine();
  size_t nHLTPath=lumiSummary.nHLTPath();
  for(unsigned int i = 0; i < nTriggers; ++i) {
    s << setw(15);
    s << lumiSummary.l1info(i).triggersource;
    
    s << setw(15);
    s << lumiSummary.l1info(i).ratecount;

    s << setw(15);
    s << lumiSummary.l1info(i).scalingfactor;
    s<<"\n";
  }
  s << setw(15) << "hltpath";
  s << setw(15) << "hltratecounter";
  s << setw(15) << "hltscaler";
  s << setw(15) << "hltinput";
  s << "\n";
  for(unsigned int i = 0; i < nHLTPath; ++i) {
    s << setw(15);
    s << lumiSummary.hltinfo(i).pathname;
    s << setw(15);
    s << lumiSummary.hltinfo(i).ratecount;
    s << setw(15);
    s << lumiSummary.hltinfo(i).scalingfactor;
    s << setw(15);
    s << lumiSummary.hltinfo(i).inputcount;
    s << "\n";
  }
  return s << "\n";
}
