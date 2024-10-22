#include "CondFormats/RunInfo/interface/LHCInfoVectorizedFields.h"
#include "CondCore/CondDB/interface/Exception.h"

LHCInfoVectorizedFields::LHCInfoVectorizedFields() : LHCInfoVectorizedFields(ISIZE, FSIZE, TSIZE, SSIZE) {}

LHCInfoVectorizedFields::LHCInfoVectorizedFields(size_t iSize, size_t fSize, size_t tSize, size_t sSize) {
  m_intParams.resize(iSize, std::vector<unsigned int>(1, 0));
  m_floatParams.resize(fSize, std::vector<float>(1, 0.));
  m_timeParams.resize(tSize, std::vector<unsigned long long>(1, 0ULL));
  m_stringParams.resize(sSize, std::vector<std::string>(1, ""));
}

template <typename T>
const T& LHCInfoVectorizedFields::getParams(const std::vector<T>& params, size_t index) {
  if (index >= params.size())
    throw cond::Exception("Parameter with index " + std::to_string(index) + " is out of range.");
  return params[index];
}

template const std::vector<unsigned int>& LHCInfoVectorizedFields::getParams(
    const std::vector<std::vector<unsigned int>>&, size_t);
template const std::vector<float>& LHCInfoVectorizedFields::getParams(const std::vector<std::vector<float>>&, size_t);
template const std::vector<unsigned long long>& LHCInfoVectorizedFields::getParams(
    const std::vector<std::vector<unsigned long long>>&, size_t);
template const std::vector<std::string>& LHCInfoVectorizedFields::getParams(
    const std::vector<std::vector<std::string>>&, size_t);

template <typename T>
T& LHCInfoVectorizedFields::accessParams(std::vector<T>& params, size_t index) {
  if (index >= params.size())
    throw cond::Exception("Parameter with index " + std::to_string(index) + " is out of range.");
  return params[index];
}

template std::vector<unsigned int>& LHCInfoVectorizedFields::accessParams(std::vector<std::vector<unsigned int>>&,
                                                                          size_t);
template std::vector<float>& LHCInfoVectorizedFields::accessParams(std::vector<std::vector<float>>&, size_t);
template std::vector<unsigned long long>& LHCInfoVectorizedFields::accessParams(
    std::vector<std::vector<unsigned long long>>&, size_t);
template std::vector<std::string>& LHCInfoVectorizedFields::accessParams(std::vector<std::vector<std::string>>&,
                                                                         size_t);

template <typename T>
const T& LHCInfoVectorizedFields::getOneParam(const std::vector<std::vector<T>>& params, size_t index) {
  if (index >= params.size())
    throw cond::Exception("Parameter with index " + std::to_string(index) + " is out of range.");
  const std::vector<T>& inner = params[index];
  if (inner.empty())
    throw cond::Exception("Parameter with index " + std::to_string(index) + " type=" + typeid(T).name() +
                          " has no value stored.");
  return inner[0];
}

template const unsigned int& LHCInfoVectorizedFields::getOneParam(const std::vector<std::vector<unsigned int>>&,
                                                                  size_t);
template const float& LHCInfoVectorizedFields::getOneParam(const std::vector<std::vector<float>>&, size_t);
template const unsigned long long& LHCInfoVectorizedFields::getOneParam(
    const std::vector<std::vector<unsigned long long>>&, size_t);
template const std::string& LHCInfoVectorizedFields::getOneParam(const std::vector<std::vector<std::string>>&, size_t);

template <typename T>
void LHCInfoVectorizedFields::setOneParam(std::vector<std::vector<T>>& params, size_t index, const T& value) {
  if (index >= params.size())
    throw cond::Exception("Parameter with index " + std::to_string(index) + " is out of range.");
  params[index] = std::vector<T>(1, value);
}

template void LHCInfoVectorizedFields::setOneParam(std::vector<std::vector<unsigned int>>& params,
                                                   size_t index,
                                                   const unsigned int& value);
template void LHCInfoVectorizedFields::setOneParam(std::vector<std::vector<float>>& params,
                                                   size_t index,
                                                   const float& value);
template void LHCInfoVectorizedFields::setOneParam(std::vector<std::vector<unsigned long long>>& params,
                                                   size_t index,
                                                   const unsigned long long& value);
template void LHCInfoVectorizedFields::setOneParam(std::vector<std::vector<std::string>>& params,
                                                   size_t index,
                                                   const std::string& value);

template <typename T>
void LHCInfoVectorizedFields::setParams(std::vector<T>& params, size_t index, const T& value) {
  if (index >= params.size())
    throw cond::Exception("Parameter with index " + std::to_string(index) + " is out of range.");
  params[index] = value;
}

template void LHCInfoVectorizedFields::setParams(std::vector<std::vector<unsigned int>>& params,
                                                 size_t index,
                                                 const std::vector<unsigned int>& value);
template void LHCInfoVectorizedFields::setParams(std::vector<std::vector<float>>& params,
                                                 size_t index,
                                                 const std::vector<float>& value);
template void LHCInfoVectorizedFields::setParams(std::vector<std::vector<unsigned long long>>& params,
                                                 size_t index,
                                                 const std::vector<unsigned long long>& value);
template void LHCInfoVectorizedFields::setParams(std::vector<std::vector<std::string>>& params,
                                                 size_t index,
                                                 const std::vector<std::string>& value);
