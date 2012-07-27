#ifndef RecoLuminosity_LumiProducer_NormDML_H 
#define RecoLuminosity_LumiProducer_NormDML_H
#include <string>
#include <vector>
#include <map>
namespace coral{
  class ISchema;
}
namespace lumi{
  class NormDML{
  public:
    class normData{
      std::string corrfunc;
      std::vector< std::pair<unsigned int,float> > afterglows;
      std::map< std::string, float > coefficientmap;
    };
    enum LumiType{HF,PIXEL};
    NormDML();
    ~NormDML(){}
    unsigned long long normIdByName(const coral::ISchema& schema,const std::string& normtagname);
    unsigned long long normIdByType(const coral::ISchema& schema,LumiType=HF,bool defaultonly=true);
    std::vector< std::pair<unsigned int,normData> >::const_iterator normById(unsigned long long normid)const;
  private:
    std::vector< std::pair<unsigned int,normData > > m_data;
  };
}//ns lumi
#endif
