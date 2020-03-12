#ifndef FIXEDARRAY_H
#define FIXEDARRAY_H

#include "CondFormats/Serialization/interface/Serializable.h"
template <typename T, unsigned int S>
class fixedArray {
public:
  fixedArray() {}

  operator T*() { return content; }
  operator const T*() const { return content; }
  T& operator[](const unsigned int i) { return content[i]; }
  const T& operator[](const unsigned int i) const { return content[i]; }

private:
  T content[S];

  COND_SERIALIZABLE;
};
#endif
