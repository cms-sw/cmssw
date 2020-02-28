#ifndef CondTools_SiStrip_SiStripDepCondObjBuilderBase_H
#define CondTools_SiStrip_SiStripDepCondObjBuilderBase_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>
#include <string>

template <typename T, typename D>
class SiStripDepCondObjBuilderBase {
public:
  SiStripDepCondObjBuilderBase(const edm::ParameterSet& pset) : _pset(pset){};
  virtual ~SiStripDepCondObjBuilderBase(){};

  virtual void initialize(){};

  /** Returns MetaData information in a stringstream */
  virtual void getMetaDataString(std::stringstream& ss){};

  /** Check MetaData information in a stringstream */
  virtual bool checkForCompatibility(std::string ss) { return true; }

  /** Returns the CondObj */
  virtual void getObj(T*& obj, const D* depObj){};

protected:
  T* obj_;
  edm::ParameterSet _pset;
};

#endif  // CondTools_SiStrip_SiStripDepCondObjBuilderBase_H
