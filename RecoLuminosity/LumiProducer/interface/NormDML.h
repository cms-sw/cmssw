#ifndef RecoLuminosity_LumiProducer_NormDML_H 
#define RecoLuminosity_LumiProducer_NormDML_H
// -*- C++ -*-
//
// Package:    LumiProducer
// Class:      NormDML
// 
/**\class NormDML NormDML.cc RecoLuminosity/LumiProducer/src/NormDML.cc

Description: 

Implementation:

*/
//
// Original Author: Zhen Xie
//       
// $Id: NormDML.h,v 1.6 2012/08/21 13:01:51 xiezhen Exp $
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
      std::string amodetag;
      unsigned int beamegev;
    };
    enum LumiType{HF,PIXEL};
    NormDML();
    ~NormDML(){}
    /*
      find the most recent id for the given norm tag 
    */
    unsigned long long normIdByName(const coral::ISchema& schema,const std::string& normtagname);
    /*
      find the most recent normid for the given norm type 
    */
    void normIdByType(const coral::ISchema& schema,std::map<std::string,unsigned long long>& resultMap,LumiType=HF,bool defaultonly=true);
    /*
      retrieve norm data by normid
    */
    void normById(const coral::ISchema& schema,
		  unsigned long long normid,
		  std::map< unsigned int,normData >&result);
    
    /*
      parse corrector functions
    */
    void parseLumiCorrector(const std::string& correctorStr,
                       std::vector<std::string>& correctorParams);
    void parseAfterglows(const std::string& afterglowStr,
			 std::map<unsigned int,float>& afterglowmap);
  };
}//ns lumi
#endif
