#include "RecoLuminosity/LumiProducer/interface/LumiCorrectionParam.h"
#include "RecoLuminosity/LumiProducer/interface/NormFunctor.h"
#include "RecoLuminosity/LumiProducer/interface/NormFunctorPluginFactory.h"
#include <iomanip>
#include <ostream>
#include <memory>
LumiCorrectionParam::LumiCorrectionParam():m_lumitype(LumiCorrectionParam::HF),m_ncollidingbx(0),m_normtag(""),m_corrfunc(""),m_amodetag("PROTPHYS"),m_beamegev(0.),m_intglumi(0.){}
LumiCorrectionParam::LumiCorrectionParam(LumiCorrectionParam::LumiType lumitype):m_lumitype(lumitype),m_ncollidingbx(0),m_normtag(""),m_corrfunc(""),m_amodetag("PROTPHYS"),m_beamegev(0.),m_intglumi(0.){}

float 
LumiCorrectionParam::getCorrection(float luminonorm)const{
  std::auto_ptr<lumi::NormFunctor> ptr(lumi::NormFunctorPluginFactory::get()->create(m_corrfunc));
  (*ptr).initialize(m_coeffmap,m_afterglows);
  float result=(*ptr).getCorrection(luminonorm,m_intglumi,m_ncollidingbx);
  return result;
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
const std::map< std::string,float >&
LumiCorrectionParam::nonlinearCoeff()const{
  return m_coeffmap;
}
const std::map< unsigned int,float >& 
LumiCorrectionParam::afterglows()const{
  return m_afterglows;
}
std::string 
LumiCorrectionParam::amodetag()const{
  return m_amodetag;
}
unsigned int 
LumiCorrectionParam::beamegev()const{
  return m_beamegev;
}
float
LumiCorrectionParam::intglumi() const{
  return m_intglumi;
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
LumiCorrectionParam::setnonlinearCoeff(std::map<std::string,float>& coeffmap){
  m_coeffmap=coeffmap;
}
void 
LumiCorrectionParam::setafterglows(std::map< unsigned int,float >& afterglows){
  m_afterglows=afterglows;
}
void 
LumiCorrectionParam::setdescription(const std::string& amodetag,unsigned int beamegev){
  m_amodetag=amodetag;
  m_beamegev=beamegev;
}
void
LumiCorrectionParam::setintglumi(float intglumi){
  m_intglumi=intglumi;
}

std::ostream& operator<<(std::ostream& s, LumiCorrectionParam const& lumiparam){
  s<<"\n LumiCorrectionParam\n";
  s<< "   normtag " << lumiparam.normtag() << "\n";
  s<< "   corrfunc " << lumiparam.corrFunc() << "\n";
  s<< "   ncollidingbx " << lumiparam.ncollidingbunches() << "\n";
  s<< "   amodetag " << lumiparam.amodetag() << "\n";
  s<< "   beamegev " << lumiparam.beamegev() << "\n";
  s<< "   intglumi " << lumiparam.intglumi() << "\n";
  std::map< std::string,float >::const_iterator it;
  std::map< std::string,float >::const_iterator itBeg=lumiparam.nonlinearCoeff().begin();
  std::map< std::string,float >::const_iterator itEnd=lumiparam.nonlinearCoeff().end();
  for(it=itBeg;it!=itEnd;++it){
    s<< "   params "<<it->first<<" "<<it->second<<"\n";
  }
  std::map< unsigned int,float >::const_iterator ait;
  std::map< unsigned int,float >::const_iterator aitBeg=lumiparam.afterglows().begin();
  std::map< unsigned int,float >::const_iterator aitEnd=lumiparam.afterglows().end();
  for(ait=aitBeg;ait!=aitEnd;++ait){
    s<< "   afterglows "<<ait->first<<" "<<ait->second<<"\n";
  }
  return s;
}
