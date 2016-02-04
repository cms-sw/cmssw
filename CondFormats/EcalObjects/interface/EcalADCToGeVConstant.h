#ifndef CondFormats_EcalObjects_EcalADCToGeVConstant_H
#define CondFormats_EcalObjects_EcalADCToGeVConstant_H
/**
 * Author: Shahram Rahatlou, University of Rome & INFN
 * Created: 22 Feb 2006
 * $Id: EcalADCToGeVConstant.h,v 1.4 2006/05/15 12:43:56 meridian Exp $
 **/

#include <iostream>

class EcalADCToGeVConstant {
  public:
    EcalADCToGeVConstant();
    EcalADCToGeVConstant(const float & EBvalue, const float & EEvalue);
    ~EcalADCToGeVConstant();
    void  setEBValue(const float& value) { EBvalue_ = value; }
    void  setEEValue(const float& value) { EEvalue_ = value; }
    float getEBValue() const { return EBvalue_; }
    float getEEValue() const { return EEvalue_; }
    void print(std::ostream& s) const {
      s << "EcalADCToGeVConstant: EB " << EBvalue_ << "; EE " << EEvalue_ << " [GeV/ADC count]";
    }
  private:
    float EBvalue_;
    float EEvalue_;
};

/**
std::ostream& operator<<(std::ostream& s, const EcalADCToGeVConstant& agc) {
  agc.print(s);
  return s;
}
**/

#endif
