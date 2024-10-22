#ifndef CondFormats_EcalObjects_EcalFunctionParameters_hh
#define CondFormats_EcalObjects_EcalFunctionParameters_hh

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>

//typedef float EcalFunParam;
typedef std::vector<float> EcalFunctionParameters;

class EcalFunParams {
public:
  EcalFunctionParameters& params() { return m_params; }
  EcalFunctionParameters const& params() const { return m_params; }

private:
  EcalFunctionParameters m_params;

  COND_SERIALIZABLE;
};

#endif
