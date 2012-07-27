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
    struct normData{
      std::string normtag;
      std::string corrfunc;
      std::map< unsigned int,float > afterglows;
      std::map< std::string, float > coefficientmap;
    };
    enum LumiType{HF,PIXEL};
    NormDML();
    ~NormDML(){}
    unsigned long long normIdByName(const coral::ISchema& schema,const std::string& normtagname);
    unsigned long long normIdByType(const coral::ISchema& schema,LumiType=HF,bool defaultonly=true);
    void normById(const coral::ISchema& schema,unsigned long long normid,std::map< unsigned int,normData >&result)const;
    
  };
}//ns lumi
#endif
