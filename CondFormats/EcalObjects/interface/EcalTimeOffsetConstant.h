#ifndef CondFormats_EcalObjects_EcalTimeOffsetConstant_H
#define CondFormats_EcalObjects_EcalTimeOffsetConstant_H
/**
 * Author: Seth Cooper, University of Minnesota
 * Created: 21 Mar 2011
 * $Id: $
 **/

#include "CondFormats/Serialization/interface/Serializable.h"

#include <iostream>

class EcalTimeOffsetConstant {
public:
  EcalTimeOffsetConstant();
  EcalTimeOffsetConstant(const float& EBvalue, const float& EEvalue);
  ~EcalTimeOffsetConstant();
  void setEBValue(const float& value) { EBvalue_ = value; }
  void setEEValue(const float& value) { EEvalue_ = value; }
  float getEBValue() const { return EBvalue_; }
  float getEEValue() const { return EEvalue_; }
  void print(std::ostream& s) const {
    s << "EcalTimeOffsetConstant: EB " << EBvalue_ << "; EE " << EEvalue_ << " [ns]";
  }

private:
  float EBvalue_;
  float EEvalue_;

  COND_SERIALIZABLE;
};

/**
std::ostream& operator<<(std::ostream& s, const EcalTimeOffsetConstant& toc) {
  toc.print(s);
  return s;
}
**/

#endif
