#ifndef RecoLuminosity_LumiProducer_NormFunctor_h
#define RecoLuminosity_LumiProducer_NormFunctor_h
#include <string>
#include <map>
namespace lumi{
  class NormFunctor{
  public:
    explicit NormFunctor();
    virtual ~NormFunctor(){}
    void initialize(const std::map< std::string , float >& coeffmap,
		    const std::map< unsigned int, float >& afterglowmap);
    virtual float getCorrection(float luminonorm,float intglumi,unsigned int nBXs)const=0;
  protected:
    std::map< std::string , float > m_coeffmap;
    std::map< unsigned int, float > m_afterglowmap;
  private:
    NormFunctor(const NormFunctor&);
    const NormFunctor& operator=(const NormFunctor&);
  };
}//ns lumi
#endif
