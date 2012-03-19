#include "RecoLuminosity/LumiProducer/interface/DIPLumiSummary.h"

#include <iomanip>
#include <ostream>
#include <iostream>
float 
DIPLumiSummary::instDelLumi() const{
  return m_instlumi;
}
float 
DIPLumiSummary::intgDelLumiByLS()const{
  return m_instlumi*23.31;
}
float 
DIPLumiSummary::intgRecLumiByLS()const{
  if(m_totdellumi<=0.0){return 0.0;}
  return m_instlumi*23.31*(m_totreclumi/m_totdellumi);
}
float 
DIPLumiSummary::intgDelLumiSinceRun()const{
  return m_totdellumi*23.31;
}
float 
DIPLumiSummary::intgRecLumiSinceRun()const{
  return m_totreclumi*23.31;
}
float 
DIPLumiSummary::deadtimefraction() const{
  if(m_totreclumi!=0){
    m_deadfrac=1.0-(m_totreclumi/m_totdellumi);
  }
  return m_deadfrac;
}
int 
DIPLumiSummary::cmsalive()const{
  return m_cmsalive;
}
std::ostream& operator<<(std::ostream& s, const DIPLumiSummary& diplumiSummary) {
  std::cout.setf(std::ios::fixed,std::ios::floatfield);
  std::cout.setf(std::ios::showpoint);
  s << "\nDumping DIPLumiSummary (/ub)\n\n";
  s << std::setw(20) << "instDelLumi = " << std::setprecision(3) << diplumiSummary.instDelLumi();
  s << std::setw(20) << "intgDelLumiByLS = " << std::setprecision(3) << diplumiSummary.intgDelLumiByLS();
  s << std::setw(20) << "intgRecLumiByLS = " << std::setprecision(3) << diplumiSummary.intgRecLumiByLS();
  s << std::setw(25) << "intgDelLumiSinceRun = " << std::setprecision(3) << diplumiSummary.intgDelLumiSinceRun();
  s << std::setw(25) << "intgRecLumiSinceRun = " << std::setprecision(3)<< diplumiSummary.intgRecLumiSinceRun();
  s << std::setw(20) << "deadtimefraction = " << std::setprecision(3) << diplumiSummary.deadtimefraction();
  s << std::setw(15) << "cmsalive = " << diplumiSummary.cmsalive();
  s << "\n";
  return s<<"\n";
}
