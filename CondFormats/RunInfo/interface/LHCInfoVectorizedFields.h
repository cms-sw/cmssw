#ifndef CondFormats_RunInfo_LHCInfoVectorizedFields_H
#define CondFormats_RunInfo_LHCInfoVectorizedFields_H

#include "CondFormats/Serialization/interface/Serializable.h"
#include <vector>

class LHCInfoVectorizedFields {
public:
  enum IntParamIndex { ISIZE = 0 };
  enum FloatParamIndex { FSIZE = 0 };
  enum TimeParamIndex { TSIZE = 0 };
  enum StringParamIndex { SSIZE = 0 };

  LHCInfoVectorizedFields();

protected:
  LHCInfoVectorizedFields(size_t iSize, size_t fSize, size_t tSize, size_t sSize);

  bool m_isData = false;
  std::vector<std::vector<unsigned int> > m_intParams;
  std::vector<std::vector<float> > m_floatParams;
  std::vector<std::vector<unsigned long long> > m_timeParams;
  std::vector<std::vector<std::string> > m_stringParams;

public:
  template <typename T>
  static const T& getParams(const std::vector<T>& params, size_t index);

  template <typename T>
  static T& accessParams(std::vector<T>& params, size_t index);

  template <typename T>
  static const T& getOneParam(const std::vector<std::vector<T> >& params, size_t index);

  template <typename T>
  static void setOneParam(std::vector<std::vector<T> >& params, size_t index, const T& value);

  template <typename T>
  static void setParams(std::vector<T>& params, size_t index, const T& value);

  COND_SERIALIZABLE;
};

#endif  // CondFormats_RunInfo_LHCInfoVectorizedFields_H