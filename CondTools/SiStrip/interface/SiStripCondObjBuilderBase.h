#ifndef CondTools_SiStrip_SiStripCondObjBuilderBase_H
#define CondTools_SiStrip_SiStripCondObjBuilderBase_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "boost/cstdint.hpp"
#include <vector>
#include <string>

template <typename T>
class SiStripCondObjBuilderBase {
  
 public:

  SiStripCondObjBuilderBase(const edm::ParameterSet& pset){};
  virtual ~SiStripCondObjBuilderBase(){};
  
  virtual void initialize(){};
			    
  /** Returns MetaData information in a stringstream */
  virtual void getMetaDataString(std::stringstream& ss){};
  
  /** Returns the CondObj */
  virtual void getObj(T* & obj){};
  
 protected:
  
  T* obj_;
  
};

#endif // CondTools_SiStrip_SiStripCondObjBuilderBase_H
