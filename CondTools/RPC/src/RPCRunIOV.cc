#include "CondTools/RPC/interface/RPCRunIOV.h"


RPCRunIOV::RPCRunIOV(const edm::EventSetup& evtSetup) 
{
  eventSetup = &evtSetup;
}


std::vector<RPCObImon::I_Item>
RPCRunIOV::getImon() {

  edm::ESHandle<RPCObImon> condRcd;
  eventSetup->get<RPCObImonRcd>().get(condRcd);
  edm::LogInfo("CondReader") << "[CondReader] Reading Cond" << std::endl;
  
  std::cout << std::endl << "=============================================" << std::endl;
  std::cout << std::endl << "==================  READER  =================" << std::endl;
  std::cout << std::endl << "=============================================" << std::endl << std::endl;
  
  
  const RPCObImon* cond = condRcd.product();
  std::vector<RPCObImon::I_Item> mycond = cond->ObImon_rpc; 
  std::vector<RPCObImon::I_Item>::iterator icond;
  
  std::cout << "--> size: " << mycond.size() << std::endl;
  
  std::cout << std::endl << "=============================================" << std::endl << std::endl;
  std::vector<RPCObImon::I_Item>::iterator first;
  first = mycond.begin();
  min = this->toUNIX(first->day, first->time);
  max = min;
  unsigned long long value;
  for(icond = mycond.begin(); icond < mycond.end(); ++icond){
    value = this->toUNIX(icond->day, icond->time);
    if (value < min) min = value;
    if (value > max) max = value;
  }
  return mycond;
}


RPCRunIOV::~RPCRunIOV(){}

bool
RPCRunIOV::isReadingNeeded(unsigned long long value)
{
  if (value < min || value > max) return true;

  return false;
}


unsigned long long 
RPCRunIOV::toDAQ(unsigned long long timeU)
{
  ::timeval tv;
  tv.tv_sec = timeU;
  tv.tv_usec = 0;
  edm::TimeValue_t daqtime=0LL;
  daqtime=tv.tv_sec;
  daqtime=(daqtime<<32)+tv.tv_usec;
  edm::Timestamp daqstamp(daqtime);
  edm::TimeValue_t dtime_ = daqstamp.value();
  unsigned long long dtime = dtime_;
  return dtime;
}


unsigned long long 
RPCRunIOV::toUNIX(int date, int time)
{
  int yea_ = (int)date/100; 
  int yea = 2000 + (date - yea_*100);
  int mon_ = (int)yea_/100;
  int mon = yea_ - mon_*100;
  int day = (int)yea_/100;
  int sec_ = (int)time/100;
  int sec = time - sec_*100;
  int min_ = (int)sec_/100;
  int min = sec_ - min_*100;
  int hou = (int)sec_/100;
  int nan = 0;
  
  coral::TimeStamp TS;  
  TS = coral::TimeStamp(yea, mon, day, hou, min, sec, nan);
  
  RPCFw* conv = new RPCFw ("","","");
  unsigned long long UT = conv->TtoUT(TS);
  
  return UT;
}




