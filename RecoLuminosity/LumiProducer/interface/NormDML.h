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
// $Id$
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
    /*
      find the most recent id for the given norm tag 
    */
    unsigned long long normIdByName(const coral::ISchema& schema,const std::string& normtagname);
    /*
      find the most recent normid for the given norm type 
    */
    unsigned long long normIdByType(const coral::ISchema& schema,LumiType=HF,bool defaultonly=true);
    /*
      retrieve norm data by normid
    */
    void normById(const coral::ISchema& schema,unsigned long long normid,std::map< unsigned int,normData >&result)const;
    
  };
}//ns lumi
#endif
