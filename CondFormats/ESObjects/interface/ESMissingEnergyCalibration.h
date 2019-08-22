#ifndef CondFormats_ESObjects_ESMissingEnergyCalibration_H
#define CondFormats_ESObjects_ESMissingEnergyCalibration_H
#include "CondFormats/Serialization/interface/Serializable.h"

#include <iostream>

class ESMissingEnergyCalibration {
public:
  ESMissingEnergyCalibration();
  ESMissingEnergyCalibration(const float& ConstAEta0,
                             const float& ConstBEta0,
                             const float& ConstAEta1,
                             const float& ConstBEta1,
                             const float& ConstAEta2,
                             const float& ConstBEta2,
                             const float& ConstAEta3,
                             const float& ConstBEta3);
  ~ESMissingEnergyCalibration();

  void setConstAEta0(const float& value) { ConstAEta0_ = value; }
  float getConstAEta0() const { return ConstAEta0_; }
  void setConstBEta0(const float& value) { ConstBEta0_ = value; }
  float getConstBEta0() const { return ConstBEta0_; }

  void setConstAEta1(const float& value) { ConstAEta1_ = value; }
  float getConstAEta1() const { return ConstAEta1_; }
  void setConstBEta1(const float& value) { ConstBEta1_ = value; }
  float getConstBEta1() const { return ConstBEta1_; }

  void setConstAEta2(const float& value) { ConstAEta2_ = value; }
  float getConstAEta2() const { return ConstAEta2_; }
  void setConstBEta2(const float& value) { ConstBEta2_ = value; }
  float getConstBEta2() const { return ConstBEta2_; }

  void setConstAEta3(const float& value) { ConstAEta3_ = value; }
  float getConstAEta3() const { return ConstAEta3_; }
  void setConstBEta3(const float& value) { ConstBEta3_ = value; }
  float getConstBEta3() const { return ConstBEta3_; }

  void print(std::ostream& s) const {
    s << "ESMissingEnergyCalibration: ES low eta constants" << ConstAEta0_ << " " << ConstBEta0_ << " / " << ConstAEta1_
      << " " << ConstBEta1_;
  }

private:
  float ConstAEta0_;
  float ConstBEta0_;

  float ConstAEta1_;
  float ConstBEta1_;

  float ConstAEta2_;
  float ConstBEta2_;

  float ConstAEta3_;
  float ConstBEta3_;

  COND_SERIALIZABLE;
};

#endif
