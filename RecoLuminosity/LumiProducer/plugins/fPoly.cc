#include "FWCore/ServiceRegistry/interface/Service.h"
#include "RecoLuminosity/LumiProducer/interface/NormFunctor.h"
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
  return luminonorm*intglumi*nBXs;
}
#include "RecoLuminosity/LumiProducer/interface/NormFunctorPluginFactory.h"
DEFINE_EDM_PLUGIN(lumi::NormFunctorPluginFactory,lumi::fPoly,"fPoly");

