#ifndef CondFormats_EcalObjects_EcalADCToGeVConstant_H
#define CondFormats_EcalObjects_EcalADCToGeVConstant_H

class EcalADCToGeVConstant {
  public:
    EcalADCToGeVConstant();
    EcalADCToGeVConstant(const float & value);
    ~EcalADCToGeVConstant();
    void  setValue(const float& value) { value_ = value; }
    float getValue() const { return value_; }

  private:
    float value_;
};
#endif
