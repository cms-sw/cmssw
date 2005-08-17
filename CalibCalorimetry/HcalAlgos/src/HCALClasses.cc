#include "CalibCalorimetry/HcalAlgos/interface/HCALClasses.h"
#include "CalibCalorimetry/HcalAlgos/interface/Frontier2.h"

namespace{
HcalDb::Gains *g_gains=0;
HcalDb::Pedestals *g_peds=0;
HcalDb::QieCMs *g_qiecms=0;
HcalDb::Ranges *g_ranges=0;
HcalDb::QLinearization *g_ql=0;
};


HcalDb::CellId::CellId(unsigned eta,unsigned phi,unsigned depth,int z)
 {
  //std::cout<<eta<<':'<<phi<<':'<<depth<<':'<<z;
  if(z!=-1 && z!=1)     throw std::logic_error("Z is out of range");
  if(!depth || depth>6) throw std::logic_error("Depth is out of range");
  if(!phi || phi>72)  throw std::logic_error("Phi is out of range");
  if(!eta || eta>41)  throw std::logic_error("Eta is out of range");
  
  int tmp=z+1;  
  id=tmp; // make it non-negative
  id|=depth<<8;
  id|=phi<<16;
  id|=eta<<24;
    
  //std::cout<<"->"<<id<<'\n';
 }


std::ostream& HcalDb::operator<<(std::ostream& o,const HcalDb::CellId &c)
 {
  o<<c.getEta()<<':'<<c.getPhi()<<':'<<c.getDepth()<<':'<<c.getZ();
  return o;
 } 



const HcalDb::Gains* HcalDb::getGains(unsigned timestamp)
 {
  if(g_gains) return g_gains;
  
  g_gains=new HcalDb::Gains();
  
  frontier2::init();
  frontier2::Request req("HCALGain","2",frontier2::BLOB);
  req.addKey("timestamp","35000");
  std::vector<const frontier2::Request*> vrq;
  vrq.push_back(&req);
  frontier2::DataSource ds;
  ds.getData(vrq);
  ds.setCurrentLoad(1);
  while(ds.next())
   {
    HcalDb::Gain v;
    unsigned eta=ds.getInt();
    unsigned phi=ds.getInt();
    unsigned depth=ds.getInt();
    int z=ds.getInt();
    v.id=HcalDb::CellId(eta,phi,depth,z);
    v.gain=ds.getFloat();
    v.error=ds.getFloat();
    g_gains->append(v);
   }
   
  return g_gains;
 }
 
 
const HcalDb::Pedestals* HcalDb::getPedestals(unsigned timestamp)
 {
  if(g_peds) return g_peds;
  
  g_peds=new HcalDb::Pedestals();
  
  frontier2::init();
  frontier2::Request req("HCALPedestals","2",frontier2::BLOB);
  req.addKey("timestamp","15000");
  std::vector<const frontier2::Request*> vrq;
  vrq.push_back(&req);
  frontier2::DataSource ds;
  ds.getData(vrq);
  ds.setCurrentLoad(1);
  while(ds.next())
   {
    HcalDb::Ped v;
    unsigned eta=ds.getInt();
    unsigned phi=ds.getInt();
    unsigned depth=ds.getInt();
    int z=ds.getInt();
    v.id=HcalDb::CellId(eta,phi,depth,z);    
    v.ped[0]=ds.getFloat();
    v.ped[1]=ds.getFloat();
    v.ped[2]=ds.getFloat();
    v.ped[3]=ds.getFloat();
    g_peds->append(v);
   }
   
  return g_peds;
 }

 
const HcalDb::QieCMs* HcalDb::getQieCMs(unsigned timestamp)
 {
  if(g_qiecms) return g_qiecms;
  
  g_qiecms=new HcalDb::QieCMs();
  
  frontier2::init();
  frontier2::Request req("HCALQieCM","1",frontier2::BLOB);
  std::vector<const frontier2::Request*> vrq;
  vrq.push_back(&req);
  frontier2::DataSource ds;
  ds.getData(vrq);
  ds.setCurrentLoad(1);
  HcalDb::QieCM v;
  while(ds.next())
   {
    unsigned eta=ds.getInt();
    unsigned phi=ds.getInt();
    unsigned depth=ds.getInt();
    int z=ds.getInt();
    v.id=HcalDb::CellId(eta,phi,depth,z);
    v.adc_channel=ds.getInt();
    v.linearized=ds.getFloat();
    v.non_linearized=ds.getFloat();
    g_qiecms->append(v);
   }
   
  return g_qiecms;
 }

  
 
const HcalDb::Ranges* HcalDb::getRanges(unsigned timestamp)
 {
  if(g_ranges) return g_ranges;
  
  g_ranges=new HcalDb::Ranges();
  
  frontier2::init();
  frontier2::Request req("HCALRange","1",frontier2::BLOB);
  std::vector<const frontier2::Request*> vrq;
  vrq.push_back(&req);
  frontier2::DataSource ds;
  ds.getData(vrq);
  ds.setCurrentLoad(1);
  while(ds.next())
   {
    HcalDb::Range v;
    unsigned eta=ds.getInt();
    unsigned phi=ds.getInt();
    unsigned depth=ds.getInt();
    int z=ds.getInt();
    v.id=HcalDb::CellId(eta,phi,depth,z);
    v.slope.reserve(16);
    v.offset.reserve(16);
    for(int i=0;i<16;i++)
     {
      v.slope.push_back(ds.getFloat());
     }
    for(int i=0;i<16;i++)
     {
      v.offset.push_back(ds.getFloat());
     }
    g_ranges->append(v);      
   }   
  return g_ranges;
 }


const HcalDb::QLinearization* HcalDb::getQLinearization(unsigned timestamp)
 {
  if(g_ql) return g_ql;

  g_ql=new HcalDb::QLinearization();

  frontier2::init();
  frontier2::Request req("HCALQLinearization","1",frontier2::BLOB);
  std::vector<const frontier2::Request*> vrq;
  vrq.push_back(&req);
  frontier2::DataSource ds;
  ds.getData(vrq);
  ds.setCurrentLoad(1);
  while(ds.next())
   {
    HcalDb::QLinearData v;
    v.bin=ds.getInt();
    v.value=ds.getFloat();
    g_ql->push_back(v);
   }
   
  return g_ql;
 }








