#include "FWCore/ServiceRegistry/interface/Service.h"
#include "RecoLuminosity/LumiProducer/interface/NormFunctor.h"
#include <algorithm>
#include <map>
namespace lumi{
  class fPoly:public NormFunctor{
  public:
    fPoly(){}
    ~fPoly(){}
    void initialize(const std::map< std::string , float >& coeffmap,
		    const std::map< unsigned int, float >& afterglowmap);
    virtual float getCorrection(float luminonorm,float intglumi,unsigned int nBXs)const;
  };
}//ns lumi

void
lumi::fPoly::initialize(const std::map< std::string , float >& coeffmap,
			const std::map< unsigned int, float >& afterglowmap){
  m_coeffmap=coeffmap;
  m_afterglowmap=afterglowmap;
}
float
lumi::fPoly::getCorrection(float luminonorm,float intglumi,unsigned int nBXs)const{
  float result=1.0;
  float avglumi=0.;
  float c1=0.;
  std::map< std::string , float >::const_iterator coeffIt=m_coeffmap.find("C1");
  if(coeffIt!=m_coeffmap.end()){
    c1=coeffIt->second;
  }
  if(c1!=0. && nBXs>0){
    avglumi=c1*luminonorm/nBXs;
  }
  float Afterglow=1.0;
  if(m_afterglowmap.size()!=0){
    std::map< unsigned int, float >::const_iterator afterglowit=--m_afterglowmap.end();
    if(nBXs>=afterglowit->first){
      Afterglow=afterglowit->second;
    }else{
      afterglowit=m_afterglowmap.upper_bound(nBXs);
      --afterglowit;
      Afterglow=afterglowit->second;
    }
  }
  float driftterm=1.0;
  coeffIt=m_coeffmap.find("DRIFT");
  if(coeffIt!=m_coeffmap.end()){
    driftterm=1.0+coeffIt->second*intglumi;
  }
  float a0=1.0;
  coeffIt=m_coeffmap.find("A0");
  if(coeffIt!=m_coeffmap.end()){
    a0=coeffIt->second;
  }
  float a1=0.;
  coeffIt=m_coeffmap.find("A1");
  if(coeffIt!=m_coeffmap.end()){
    a1=coeffIt->second;
  }
  float a2=0.;
  coeffIt=m_coeffmap.find("A2");
  if(coeffIt!=m_coeffmap.end()){
    a2=coeffIt->second;
  }
  result=a0*Afterglow/(1.+a1*avglumi+a2*avglumi*avglumi)*driftterm;
  return result;
}
#include "RecoLuminosity/LumiProducer/interface/NormFunctorPluginFactory.h"
DEFINE_EDM_PLUGIN(lumi::NormFunctorPluginFactory,lumi::fPoly,"fPoly");

