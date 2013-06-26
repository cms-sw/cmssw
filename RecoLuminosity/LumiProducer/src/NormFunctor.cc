#include "RecoLuminosity/LumiProducer/interface/NormFunctor.h"
lumi::NormFunctor::NormFunctor(){}
void lumi::NormFunctor::initialize(const std::map< std::string , float >& coeffmap,const std::map< unsigned int, float >& afterglowmap){
  m_coeffmap=coeffmap;
  m_afterglowmap=afterglowmap;
}

