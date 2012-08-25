#include "CondTools/RPC/interface/RPCRunIOV.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include "DataFormats/Provenance/interface/Timestamp.h"
#include <sys/time.h>


namespace 
{
  std::string toString (int i)
  {
    char temp[20];
    sprintf (temp, "%d", i);
    return ((std::string) temp);
  }
}



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

  if (mycond.size() == 0) {
    min_I = 0;
    max_I = 0;
    return mycond;
  }

  std::vector<RPCObImon::I_Item>::iterator first;
  first = mycond.begin();
  min_I = this->toUNIX(first->day, first->time);
  max_I = min_I;
  unsigned long long value;
  for(icond = mycond.begin(); icond < mycond.end(); ++icond){
    value = this->toUNIX(icond->day, icond->time);
    if (value < min_I) min_I = value;
    if (value > max_I) max_I = value;
  }
  return mycond;
}



std::vector<RPCObVmon::V_Item>
RPCRunIOV::getVmon() {

  edm::ESHandle<RPCObVmon> condRcd;
  eventSetup->get<RPCObVmonRcd>().get(condRcd);
   
  std::cout << std::endl << "=============================================" << std::endl;
  std::cout << std::endl << "===============  VMON READER  ===============" << std::endl;
  std::cout << std::endl << "=============================================" << std::endl << std::endl;
  
  
  const RPCObVmon* cond = condRcd.product();
  std::vector<RPCObVmon::V_Item> mycond = cond->ObVmon_rpc; 
  std::vector<RPCObVmon::V_Item>::iterator icond;
  
  std::cout << ">>> Object VMON" << std::endl;
  std::cout << "    size " << mycond.size() << std::endl;
  
  std::cout << std::endl << "=============================================" << std::endl << std::endl;

  if (mycond.size() == 0) {
    min_I = 0;
    max_I = 0;
    return mycond;
  }

  std::vector<RPCObVmon::V_Item>::iterator first;
  first = mycond.begin();
  min_V = this->toUNIX(first->day, first->time);
  max_V = min_I;
  unsigned long long value;
  for(icond = mycond.begin(); icond < mycond.end(); ++icond){
    value = this->toUNIX(icond->day, icond->time);
    if (value < min_V) min_V = value;
    if (value > max_V) max_V = value;
  }
  return mycond;
}



std::vector<RPCObTemp::T_Item>
RPCRunIOV::getTemp() {

  edm::ESHandle<RPCObTemp> condRcd;
  eventSetup->get<RPCObTempRcd>().get(condRcd);

  std::cout << std::endl << "=============================================" << std::endl;
  std::cout << std::endl << "===============  TEMP READER  ===============" << std::endl;
  std::cout << std::endl << "=============================================" << std::endl << std::endl;


  const RPCObTemp* cond = condRcd.product();
  std::vector<RPCObTemp::T_Item> mycond = cond->ObTemp_rpc;
  std::vector<RPCObTemp::T_Item>::iterator icond;

  std::cout << ">>> Object TEMPERATURE" << std::endl;
  std::cout << "    size " << mycond.size() << std::endl;

  std::cout << std::endl << "=============================================" << std::endl << std::endl;
  
  if (mycond.size() == 0) {
    min_I = 0;
    max_I = 0;
    return mycond;
  }
  
  std::vector<RPCObTemp::T_Item>::iterator first;
  first = mycond.begin();
  min_T = this->toUNIX(first->day, first->time);
  max_T = min_T;
  unsigned long long value;
  for(icond = mycond.begin(); icond < mycond.end(); ++icond){
    value = this->toUNIX(icond->day, icond->time);
    if (value < min_T) min_T = value;
    if (value > max_T) max_T = value;
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

  RPCObPVSSmap::Item pvssItem={0,0,0,0,0,0,0,0,0};
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
RPCRunIOV::DAQtoUNIX(unsigned long long *time)
{
  timeval *tmval=(timeval*)time;
  unsigned long long int curVal=(tmval->tv_usec*1000000LL)+tmval->tv_sec;
  return curVal;
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
  RPCFw conv ("","","");
  unsigned long long UT = conv.TtoUT(TS);
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



//-----------chamber Name -------------------------------------
std::string
RPCRunIOV::chamberName(chRAW ch){

  using namespace std;
  string chambername, sector, station, DP, ring;

  // BARREL
  if (ch.region == 0) {
    switch(ch.ring) {
    case 2:  chambername = "WP2";
    case 1:  chambername = "WP1";
    case 0:  chambername = "W00";
    case -1: chambername = "WM1";
    case -2: chambername = "WM2";
    }
    sector  = toString (ch.sector);
    station = toString (ch.station);
    chambername  += "_S"+sector+"_RB"+station;
        
    switch(ch.station) {
    case 1:; case 2:  
      if (ch.subsector == 1) chambername += "minus";
      if (ch.subsector == 2) chambername += "minus";
    case 3:    
      if(ch.layer == 1)chambername += "in";
      if(ch.layer == 2)chambername += "out";
    case 4:
      if(ch.sector != 9 && ch.sector != 11) {
	if (ch.subsector == 1) chambername += "minusminus";
	if (ch.subsector == 2) chambername += "minus";
	if (ch.subsector == 3) chambername += "plus";
	if (ch.subsector == 4) chambername += "plusplus";
      } else {
	if (ch.subsector == 1) chambername += "minus";
	if (ch.subsector == 2) chambername += "minus";
      }
    }
  }
  // ENDCAP
  else{
    int DP_ = 6*(ch.sector-1)+ch.subsector;
    DP      = toString (DP_);
    ring    = toString (ch.ring);
    station = toString (ch.station);
    if (ch.region == 1) chambername += "DP";
    if (ch.region == -1)chambername += "DM";
    chambername += station+"_R"+ring+"_C"+DP;
  }
  return chambername;
}
