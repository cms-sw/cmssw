#ifndef CondFormats_ESObjects_ESEEIntercalibConstants_H
#define CondFormats_ESObjects_ESEEIntercalibConstants_H
#include <iostream>

class ESEEIntercalibConstants {

  public:

    ESEEIntercalibConstants();
    ESEEIntercalibConstants(const float & gammaLow, const float & alphaLow, const float & gammaHigh, const float & alphaHigh );
    ~ESEEIntercalibConstants();

    void  setGammaLow(const float& value) { gammaLow_ = value; }
    float getGammaLow() const { return gammaLow_; }
    void  setAlphaLow(const float& value) { alphaLow_ = value; }
    float getAlphaLow() const { return alphaLow_; }

    void  setGammaHigh(const float& value) { gammaHigh_ = value; }
    float getGammaHigh() const { return gammaHigh_; }
    void  setAlphaHigh(const float& value) { alphaHigh_ = value; }
    float getAlphaHigh() const { return alphaHigh_; }

    void print(std::ostream& s) const {
      s << "ESEEIntercalibConstants: ES low gain (gamma, alpha) / high gain (gamma, alpha)" << gammaLow_ << " " << alphaLow_<< " / " << gammaHigh_ <<" "<<alphaHigh_;
    }

  private:

    float gammaLow_;
    float alphaLow_;
    float gammaHigh_;
    float alphaHigh_;
};

#endif
