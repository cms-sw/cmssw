#ifndef CondFormats_ESObjects_ESEEIntercalibConstants_H
#define CondFormats_ESObjects_ESEEIntercalibConstants_H
#include "CondFormats/Serialization/interface/Serializable.h"

#include <iostream>

class ESEEIntercalibConstants {
public:
  ESEEIntercalibConstants();
  ESEEIntercalibConstants(const float& gammaLow0,
                          const float& alphaLow0,
                          const float& gammaHigh0,
                          const float& alphaHigh0,
                          const float& gammaLow1,
                          const float& alphaLow1,
                          const float& gammaHigh1,
                          const float& alphaHigh1,
                          const float& gammaLow2,
                          const float& alphaLow2,
                          const float& gammaHigh2,
                          const float& alphaHigh2,
                          const float& gammaLow3,
                          const float& alphaLow3,
                          const float& gammaHigh3,
                          const float& alphaHigh3);
  ~ESEEIntercalibConstants();

  void setGammaLow0(const float& value) { gammaLow0_ = value; }
  float getGammaLow0() const { return gammaLow0_; }
  void setAlphaLow0(const float& value) { alphaLow0_ = value; }
  float getAlphaLow0() const { return alphaLow0_; }

  void setGammaLow1(const float& value) { gammaLow1_ = value; }
  float getGammaLow1() const { return gammaLow1_; }
  void setAlphaLow1(const float& value) { alphaLow1_ = value; }
  float getAlphaLow1() const { return alphaLow1_; }

  void setGammaLow2(const float& value) { gammaLow2_ = value; }
  float getGammaLow2() const { return gammaLow2_; }
  void setAlphaLow2(const float& value) { alphaLow2_ = value; }
  float getAlphaLow2() const { return alphaLow2_; }

  void setGammaLow3(const float& value) { gammaLow3_ = value; }
  float getGammaLow3() const { return gammaLow3_; }
  void setAlphaLow3(const float& value) { alphaLow3_ = value; }
  float getAlphaLow3() const { return alphaLow3_; }

  void setGammaHigh0(const float& value) { gammaHigh0_ = value; }
  float getGammaHigh0() const { return gammaHigh0_; }
  void setAlphaHigh0(const float& value) { alphaHigh0_ = value; }
  float getAlphaHigh0() const { return alphaHigh0_; }

  void setGammaHigh1(const float& value) { gammaHigh1_ = value; }
  float getGammaHigh1() const { return gammaHigh1_; }
  void setAlphaHigh1(const float& value) { alphaHigh1_ = value; }
  float getAlphaHigh1() const { return alphaHigh1_; }

  void setGammaHigh2(const float& value) { gammaHigh2_ = value; }
  float getGammaHigh2() const { return gammaHigh2_; }
  void setAlphaHigh2(const float& value) { alphaHigh2_ = value; }
  float getAlphaHigh2() const { return alphaHigh2_; }

  void setGammaHigh3(const float& value) { gammaHigh3_ = value; }
  float getGammaHigh3() const { return gammaHigh3_; }
  void setAlphaHigh3(const float& value) { alphaHigh3_ = value; }
  float getAlphaHigh3() const { return alphaHigh3_; }

  void print(std::ostream& s) const {
    s << "ESEEIntercalibConstants: ES low gain (gamma, alpha) / high gain (gamma, alpha)" << gammaLow0_ << " "
      << alphaLow0_ << " / " << gammaHigh0_ << " " << alphaHigh0_;
  }

private:
  // both planes work perfectly
  float gammaLow0_;
  float alphaLow0_;
  float gammaHigh0_;
  float alphaHigh0_;

  // both planes do not work at all
  float gammaLow1_;
  float alphaLow1_;
  float gammaHigh1_;
  float alphaHigh1_;

  // only the first plane works
  float gammaLow2_;
  float alphaLow2_;
  float gammaHigh2_;
  float alphaHigh2_;

  // only the second plane works
  float gammaLow3_;
  float alphaLow3_;
  float gammaHigh3_;
  float alphaHigh3_;

  COND_SERIALIZABLE;
};

#endif
