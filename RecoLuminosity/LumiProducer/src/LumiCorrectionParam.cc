#include "RecoLuminosity/LumiProducer/interface/LumiCorrectionParam.h"
#include <iomanip>
#include <ostream>
LumiCorrectionParam::LumiCorrectionParam():m_lumitype(LumiCorrectionParam::HF),m_ncollidingbx(0){}
LumiCorrectionParam::LumiCorrectionParam(LumiCorrectionParam::LumiType lumitype):m_lumitype(lumitype),m_ncollidingbx(0){}
void
LumiCorrectionParam::setNBX(unsigned int nbx){
  m_ncollidingbx=nbx;
}
unsigned int 
LumiCorrectionParam::ncollidingbunches()const{
  return m_ncollidingbx;
}
std::ostream& operator<<(std::ostream& s, LumiCorrectionParam const& lumiparam){
  s<<"\n LumiCorrectionParam\n";
  s << std::setw(12) << "ncollidingbx " << lumiparam.ncollidingbunches() << "\n";
  return s;
}
