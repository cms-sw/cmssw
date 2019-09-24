#ifndef CondTools_SiStrip_SiStripCondObjBuilderBase_H
#define CondTools_SiStrip_SiStripCondObjBuilderBase_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>
#include <string>

template <typename T>
class SiStripCondObjBuilderBase {
public:
  SiStripCondObjBuilderBase(const edm::ParameterSet& pset) : _pset(pset){};
  virtual ~SiStripCondObjBuilderBase() noexcept(false){};

  virtual void initialize(){};

  /** Returns MetaData information in a stringstream */
  virtual void getMetaDataString(std::stringstream& ss){};

  /** Check MetaData information in a stringstream */
  virtual bool checkForCompatibility(std::string ss) { return true; }

  /** Returns the CondObj */
  virtual void getObj(T*& obj){};

protected:
  T* obj_;
  edm::ParameterSet _pset;
};

#endif  // CondTools_SiStrip_SiStripCondObjBuilderBase_H
