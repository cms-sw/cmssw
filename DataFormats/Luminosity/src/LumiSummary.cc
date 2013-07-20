
// $Id: LumiSummary.cc,v 1.27 2012/09/20 14:08:11 xiezhen Exp $

#include "DataFormats/Luminosity/interface/LumiSummary.h"

#include <iomanip>
#include <ostream>
#include <iostream>
float
LumiSummary::avgInsDelLumi()const{ 
  size_t iIndex=lumiversion_.rfind("v");
  if(iIndex != std::string::npos){//i.e. not "-1" and not "DIP", "-1" and "DIP" lumi are already corrected and unit conversion included in the raw data. 
    return avginsdellumi_*1000.0;
  }
  return avginsdellumi_;
}
float
LumiSummary::intgDelLumi()const{
  return this->avgInsDelLumi()*float(this->lumiSectionLength());
}
float
LumiSummary::avgInsDelLumiErr()const{ 
  return  avginsdellumierr_;
}
float 
LumiSummary::intgRecLumi()const{
  return this->avgInsRecLumi()*float(this->lumiSectionLength());
}
short
LumiSummary::lumiSecQual()const {
  return lumisecqual_; 
}
unsigned long long
LumiSummary::deadcount() const{
  return deadcount_;
}
unsigned long long
LumiSummary::bitzerocount() const{
  return bitzerocount_;
}
float 
LumiSummary::deadFrac() const {
  //definition: deadcount/bitzerocount
  //if no trigger data, return deadfraction 1.0,mask out this LS
  //if bitzerocount=0, return -1.0 meaning no beam
  if(lumiversion_=="DIP"){
    return float(deadcount_)/float(bitzerocount_);
  }
  if (l1data_.size()==0) return 1.0;
  if (bitzerocount_==0) return -1.0;
  return float(deadcount_)/float(bitzerocount_);
}
float 
LumiSummary::liveFrac() const { 
  //1-deadfraction
  //else if deadfraction<0 meaning no beam, live fraction=0
  //
  if (deadFrac()<0) return 0;
  return 1-deadFrac();
}
float
LumiSummary::lumiSectionLength() const {
  //numorbits*3564*24.95e-09
  return numorbit_*3564.0*24.95e-9;
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
  return avginsdellumi_>0;
}
LumiSummary::L1  
LumiSummary::l1info(unsigned int idx)const{
  return l1data_.at(idx);
}
LumiSummary::HLT  
LumiSummary::hltinfo(unsigned int idx) const {
  return hltdata_.at(idx);
}
size_t
LumiSummary::nTriggerLine()const{
  return l1data_.size();
}
size_t
LumiSummary::nHLTPath()const{
  return hltdata_.size();
}
float
LumiSummary::avgInsRecLumi() const {
  return this->avgInsDelLumi() * liveFrac(); 
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
LumiSummary::setDeadCount(unsigned long long deadcount){
  deadcount_=deadcount;
}
void 
LumiSummary::setBitZeroCount(unsigned long long bitzerocount){
  bitzerocount_=bitzerocount;
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
  s << "  avgIntgDelLumi = " << lumiSummary.intgDelLumi() <<"\n";
  s << "  avgInsDelLumiErr = " << lumiSummary.avgInsDelLumiErr() << "\n";
  s << "  lumiSecQual = " << lumiSummary.lumiSecQual() << "\n";
  s << "  deadCount = " << lumiSummary.deadcount() << "\n";
  s << "  bitZeroCount = " << lumiSummary.bitzerocount() << "\n";
  s << "  deadFrac = " << (float)lumiSummary.deadFrac() << "\n";
  s << "  liveFrac = " << (float)lumiSummary.liveFrac() << "\n";
  s << "  lsNumber = " << lumiSummary.lsNumber() << "\n";
  s << "  startOrbit = " << lumiSummary.startOrbit() <<"\n";
  s << "  numOrbit = " << lumiSummary.numOrbit() <<"\n";
  s << "  avgInsRecLumi = " << lumiSummary.avgInsRecLumi() << "\n";
  s << "  avgInsRecLumiErr = "  << lumiSummary.avgInsRecLumiErr() << "\n\n";
  s << std::setw(15) << "l1nameidx";
  s << std::setw(15) << "l1prescale";
  s << "\n";
  size_t nTriggers=lumiSummary.nTriggerLine();
  size_t nHLTPath=lumiSummary.nHLTPath();
  for(unsigned int i = 0; i < nTriggers; ++i) {
    s << std::setw(15);
    s << lumiSummary.l1info(i).triggernameidx;
    s << std::setw(15);
    s << lumiSummary.l1info(i).prescale;
    s<<"\n";
  }
  s << std::setw(15) << "hltpathidx";
  s << std::setw(15) << "hltprescale";
  s << "\n";
  for(unsigned int i = 0; i < nHLTPath; ++i) {
    s << std::setw(15);
    s << lumiSummary.hltinfo(i).pathnameidx;
    s << std::setw(15);
    s << lumiSummary.hltinfo(i).prescale;
    s << "\n";
  }
  return s << "\n";
}
