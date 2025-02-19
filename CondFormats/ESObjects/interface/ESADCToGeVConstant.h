#ifndef CondFormats_ESObjects_ESADCToGeVConstant_H
#define CondFormats_ESObjects_ESADCToGeVConstant_H
#include <iostream>

class ESADCToGeVConstant {
  public:
    ESADCToGeVConstant();
    ESADCToGeVConstant(const float & ESvaluelow, const float & ESvaluehigh);
    ~ESADCToGeVConstant();
    void  setESValueLow(const float& value) { ESvaluelow_ = value; }
    float getESValueLow() const { return ESvaluelow_; }
    void  setESValueHigh(const float& value) { ESvaluehigh_ = value; }
    float getESValueHigh() const { return ESvaluehigh_; }
    void print(std::ostream& s) const {
      s << "ESADCToGeVConstant: ES low/high " << ESvaluelow_ << " / " << ESvaluehigh_ <<" [GeV/ADC count]";
    }
  private:
    float ESvaluelow_;
    float ESvaluehigh_;
};


#endif
