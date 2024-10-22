#ifndef CondFormats_ESObjects_ESGain_H
#define CondFormats_ESObjects_ESGain_H
#include "CondFormats/Serialization/interface/Serializable.h"

#include <iostream>

class ESGain {
public:
  ESGain();
  ESGain(const float& gain);
  ~ESGain();
  void setESGain(const float& value) { gain_ = value; }
  float getESGain() const { return gain_; }
  void print(std::ostream& s) const { s << "ESGain: " << gain_; }

private:
  float gain_;

  COND_SERIALIZABLE;
};

#endif
