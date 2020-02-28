#include "CondFormats/CSCObjects/interface/CSCDBL1TPParametersExtended.h"

CSCDBL1TPParametersExtended::CSCDBL1TPParametersExtended() {
  paramsInt_.resize(paramNamesInt_.size());
  paramsBool_.resize(paramNamesBool_.size());
}

CSCDBL1TPParametersExtended::~CSCDBL1TPParametersExtended() {}

int CSCDBL1TPParametersExtended::getValueInt(const std::string& s) const {
  const int index = find(paramNamesInt_.begin(), paramNamesInt_.end(), s) - paramNamesInt_.begin();
  return paramsInt_[index];
}

bool CSCDBL1TPParametersExtended::getValueBool(const std::string& s) const {
  const int index = find(paramNamesBool_.begin(), paramNamesBool_.end(), s) - paramNamesBool_.begin();
  return paramsBool_[index];
}

void CSCDBL1TPParametersExtended::setValue(const std::string& s, int v) {
  const int index = find(paramNamesInt_.begin(), paramNamesInt_.end(), s) - paramNamesInt_.begin();
  paramsInt_[index] = v;
}

void CSCDBL1TPParametersExtended::setValue(const std::string& s, bool v) {
  const int index = find(paramNamesBool_.begin(), paramNamesBool_.end(), s) - paramNamesBool_.begin();
  paramsBool_[index] = v;
}
