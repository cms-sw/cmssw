#include "CondTools/RPC/interface/RPCRunIOV.h"


RPCRunIOV::RPCRunIOV()
{}


RPCRunIOV::RPCRunIOV(const edm::EventSetup& evtSetup) 
{
  eventSetup = &evtSetup;
}


std::vector<RPCObImon::I_Item>
RPCRunIOV::getImon() {

  edm::ESHandle<RPCObImon> condRcd;
  eventSetup->get<RPCObImonRcd>().get(condRcd);
   
  std::cout << std::endl << "=============================================" << std::endl;
  std::cout << std::endl << "===============  IMON READER  ===============" << std::endl;
  std::cout << std::endl << "=============================================" << std::endl << std::endl;
  
  
  const RPCObImon* cond = condRcd.product();
  std::vector<RPCObImon::I_Item> mycond = cond->ObImon_rpc; 
  std::vector<RPCObImon::I_Item>::iterator icond;
  
  std::cout << ">>> Object IMON" << std::endl;
  std::cout << "    size " << mycond.size() << std::endl;
  
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



std::map<int, RPCObPVSSmap::Item>
RPCRunIOV::getPVSSMap()
{

  edm::ESHandle<RPCObPVSSmap> pvssRcd;
  eventSetup->get<RPCObPVSSmapRcd>().get(pvssRcd);
  
  std::cout << std::endl << "=============================================" << std::endl;
  std::cout << std::endl << "===============  PVSS READER  ===============" << std::endl;
  std::cout << std::endl << "=============================================" << std::endl << std::endl;

  const RPCObPVSSmap* pvss = pvssRcd.product();
  std::vector<RPCObPVSSmap::Item> mypvss = pvss->ObIDMap_rpc;
  std::vector<RPCObPVSSmap::Item>::iterator ipvss;

  std::cout << ">>> Object PVSS" << std::endl;
  std::cout << "    size " << mypvss.size() << std::endl;

  RPCObPVSSmap::Item pvssItem;
  int id;
  std::map<int, RPCObPVSSmap::Item> pvssmap;
  for(ipvss = mypvss.begin(); ipvss < mypvss.end(); ++ipvss){
    id = ipvss->dpid;
    pvssItem.region = ipvss->region;
    pvssItem.ring = ipvss->ring;
    pvssItem.station = ipvss->station;
    pvssItem.sector = ipvss->sector;
    pvssItem.layer = ipvss->layer;
    pvssItem.subsector = ipvss->subsector;
    pvssItem.suptype = ipvss->suptype;
    pvssmap.insert ( std::pair<int, RPCObPVSSmap::Item>(id, pvssItem) );

  }

  std::cout << std::endl << "=============================================" << std::endl << std::endl;

  return pvssmap;
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



// this methos filters data
std::vector<RPCObImon::I_Item>
RPCRunIOV::filterIMON(std::vector<RPCObImon::I_Item> imon, unsigned long long since, unsigned long long till)
{

  std::cout << std::endl << "=============================================" << std::endl;
  std::cout << std::endl << "============    FILTERING DATA    ===========" << std::endl;
  std::cout << std::endl << "=============================================" << std::endl << std::endl;
  std::vector<RPCObImon::I_Item>::iterator it;
  RPCFw conv ("","","");
  int n = 0;
  for ( it=imon.begin(); it < imon.end(); it++ ) {
    n++;
    int day = (int)it->day/10000;
    int mon = (int)(it->day - day*10000)/100;
    int yea = (int)(it->day - day*10000 - mon*100)+2000;
    int hou = (int)it->time/10000;
    int min = (int)(it->time - hou*10000)/100;
    int sec = (int)(it->time - hou*10000 - min*100);
    int nan = 0;
    coral::TimeStamp timeD = coral::TimeStamp(yea, mon, day, hou, min, sec, nan);
    unsigned long long timeU = conv.TtoUT(timeD);
    if (timeU < till && timeU > since) filtImon.push_back(*it);
  }
  return filtImon;
}



