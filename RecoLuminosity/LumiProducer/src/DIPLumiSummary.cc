#include "RecoLuminosity/LumiProducer/interface/DIPLumiSummary.h"

#include <iomanip>
#include <ostream>
#include <iostream>
bool 
DIPLumiSummary::isNull()const{
  if(m_runnum==0 && m_ls==0){
    return true;
  }
  return false;
}
float 
DIPLumiSummary::instDelLumi() const{
  return m_instlumi;
}
float 
DIPLumiSummary::intgDelLumiByLS()const{
  if(m_dellumi<=0.0){return 0.0;}
  return m_dellumi;
}
float 
DIPLumiSummary::intgRecLumiByLS()const{
  if(m_reclumi<=0.0){return 0.0;}
  return m_reclumi;
}
float 
DIPLumiSummary::deadtimefraction() const{
  if(m_reclumi>0.0){
    m_deadfrac=1.0-(m_reclumi/m_dellumi);
  }
  return m_deadfrac;
}
int 
DIPLumiSummary::cmsalive()const{
  return m_cmsalive;
}
unsigned int 
DIPLumiSummary::fromRun()const{
  return m_runnum;
}
/**
   from which ls data come from
**/
unsigned int 
DIPLumiSummary::fromLS()const{
  return m_ls;
}
void 
DIPLumiSummary::setOrigin(unsigned int runnumber,unsigned int ls){
  m_runnum=runnumber;
  m_ls=ls;
}
void setOrigin(unsigned int runnumber,unsigned int ls);
std::ostream& operator<<(std::ostream& s, const DIPLumiSummary& diplumiSummary) {
  std::cout.setf(std::ios::fixed,std::ios::floatfield);
  std::cout.setf(std::ios::showpoint);
  s << "\nDumping DIPLumiSummary (/ub)\n\n";
  s << std::setw(20) << "instDelLumi = " << std::setprecision(3) << diplumiSummary.instDelLumi();
  s << std::setw(20) << "intgDelLumiByLS = " << std::setprecision(3) << diplumiSummary.intgDelLumiByLS();
  s << std::setw(20) << "intgRecLumiByLS = " << std::setprecision(3) << diplumiSummary.intgRecLumiByLS();
  s << std::setw(20) << "deadtimefraction = " << std::setprecision(3) << diplumiSummary.deadtimefraction();
  s << std::setw(15) << "cmsalive = " << diplumiSummary.cmsalive();
  s << "\n";
  return s<<"\n";
}
