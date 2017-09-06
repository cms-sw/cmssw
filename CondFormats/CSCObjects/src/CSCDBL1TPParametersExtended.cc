#include "CondFormats/CSCObjects/interface/CSCDBL1TPParametersExtended.h"

CSCDBL1TPParametersExtended::CSCDBL1TPParametersExtended()
{
  params_.resize(paramNames_.size());

}

CSCDBL1TPParametersExtended::~CSCDBL1TPParametersExtended()
{
}

S CSCDBL1TPParametersExtended::getValue(const std::string& s) const
{
  const int index = find(paramNames_.begin(), paramNames_.end(), s) - paramNames_.begin();
  return params_[index];
}

void CSCDBL1TPParametersExtended::setValue(const std::string& s, const S& v)
{
  const int index = find(paramNames_.begin(), paramNames_.end(), s) - paramNames_.begin();
  params_[index] = v;
}
