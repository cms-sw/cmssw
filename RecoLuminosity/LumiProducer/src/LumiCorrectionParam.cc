#include "RecoLuminosity/LumiProducer/interface/LumiCorrectionParam.h"
#include <iomanip>
#include <ostream>
LumiCorrectionParam::LumiCorrectionParam():m_lumitype(LumiCorrectionParam::HF),m_ncollidingbx(0),m_normtag(""),m_corrfunc(""),m_amodetag("PROTPHYS"),m_beamegev(0.){}
LumiCorrectionParam::LumiCorrectionParam(LumiCorrectionParam::LumiType lumitype):m_lumitype(lumitype),m_ncollidingbx(0),m_normtag(""),m_corrfunc(""),m_amodetag("PROTPHYS"),m_beamegev(0.){}

float 
LumiCorrectionParam::getCorrection()const{
  return 1.0;
}
unsigned int 
LumiCorrectionParam::ncollidingbunches()const{
  return m_ncollidingbx;
}
std::string 
LumiCorrectionParam::normtag()const{
  return m_normtag;
}
std::string 
LumiCorrectionParam::corrFunc()const{
  return m_corrfunc;
}
std::map<const std::string,float>::const_iterator 
LumiCorrectionParam::nonlinearCoeff()const{
  return m_coeffmap.begin();
}
std::vector< std::pair<unsigned int,float> >::const_iterator 
LumiCorrectionParam::afterglows()const{
  return m_afterglows.begin();
}
std::string 
LumiCorrectionParam::amodetag()const{
  return m_amodetag;
}
unsigned int 
LumiCorrectionParam::beamegev()const{
  return m_beamegev;
}
void 
LumiCorrectionParam::setNBX(unsigned int nbx){
  m_ncollidingbx=nbx;
}
void 
LumiCorrectionParam::setNormtag(const std::string& normtag){
  m_normtag=normtag;
}
void 
LumiCorrectionParam::setcorrFunc(const std::string& corrfunc){
  m_corrfunc=corrfunc;
}
void 
LumiCorrectionParam::setnonlinearCoeff(std::map<const std::string,float>& coeffmap){
  m_coeffmap=coeffmap;
}
void 
LumiCorrectionParam::setafterglows(std::vector< std::pair<unsigned int,float> >& afterglows){
  m_afterglows=afterglows;
}

std::ostream& operator<<(std::ostream& s, LumiCorrectionParam const& lumiparam){
  s<<"\n LumiCorrectionParam\n";
  s << std::setw(12) << "ncollidingbx " << lumiparam.ncollidingbunches() << "\n";
  return s;
}
