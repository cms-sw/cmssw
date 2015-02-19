#ifndef DytParamsObject_h
#define DytParamsObject_h

#include <vector>
#include "CondFormats/RecoMuonObjects/interface/DYTParamObject.h"
#include "CondFormats/Serialization/interface/Serializable.h"

class DYTParamsObject {
 public:

  DYTParamsObject() {};
  ~DYTParamsObject() { m_paramObjs.clear(); };

  // Add a parameter to the vector of parameters
  void addParamObject(const DYTParamObject & obj) { m_paramObjs.push_back(obj); };

  // Set the parametrized formula                                                                                                                                                                            
  void  setFormula(std::string formula) { m_formula = formula; };

  // Get the list of parameters
  const std::vector<DYTParamObject> & getParamObjs() const { return m_paramObjs; };

  // Get the functional parametrization                                                                                                                                                                    
  const std::string & formula() const { return m_formula; };

 private:
  
  std::vector<DYTParamObject> m_paramObjs;

  std::string m_formula;

  COND_SERIALIZABLE;
};

#endif
