#ifndef CondFormats_EcalObjects_EcalADCToGeVConstant_H
#define CondFormats_EcalObjects_EcalADCToGeVConstant_H

#include <iostream>

class EcalADCToGeVConstant {
  public:
    EcalADCToGeVConstant();
    EcalADCToGeVConstant(const float & value);
    ~EcalADCToGeVConstant();
    void  setValue(const float& value) { value_ = value; }
    float getValue() const { return value_; }
    void print(std::ostream& s) const {
      s << "EcalADCToGeVConstant: " << value_ << " GeV/ADC count";
    }
  private:
    float value_;
};

/**
std::ostream& operator<<(std::ostream& s, const EcalADCToGeVConstant& agc) {
  agc.print(s);
  return s;
}
**/

#endif
