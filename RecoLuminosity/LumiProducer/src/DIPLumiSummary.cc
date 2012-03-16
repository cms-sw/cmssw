#include "RecoLuminosity/LumiProducer/interface/DIPLumiSummary.h"

#include <iomanip>
#include <ostream>
#include <iostream>
float 
DIPLumiSummary::instDelLumi() const{
  return m_instlumi;
}
float 
DIPLumiSummary::intgDelLumi()const{
  return m_dellumi;
}
float 
DIPLumiSummary::intgRecLumi()const{
  return m_reclumi;
}
float 
DIPLumiSummary::deadtimefraction() const{
  return m_deadfrac;
}
int 
DIPLumiSummary::cmsalive()const{
  return m_cmsalive;
}
std::ostream& operator<<(std::ostream& s, const DIPLumiSummary& diplumiSummary) {
  s << "\nDumping DIPLumiSummary\n\n";
  s << std::setw(15) << "  instDelLumi = " << diplumiSummary.instDelLumi();
  s << std::setw(15) << "  intgDelLumi = " << diplumiSummary.intgDelLumi();
  s << std::setw(15) << "  intgRecLumi = " << diplumiSummary.intgRecLumi();
  s << std::setw(15) << "  deadtimefraction = " << diplumiSummary.deadtimefraction();
  s << std::setw(15) << "  cmsalive = " << diplumiSummary.cmsalive();
  s << "\n";
  return s<<"\n";
}
