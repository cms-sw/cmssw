#ifndef CondFormats_ESObjects_ESMIPToGeVConstant_H
#define CondFormats_ESObjects_ESMIPToGeVConstant_H
#include <iostream>

class ESMIPToGeVConstant {

  public:

    ESMIPToGeVConstant();
    ESMIPToGeVConstant(const float & ESvaluelow, const float & ESvaluehigh);
    ~ESMIPToGeVConstant();
    void  setESValueLow(const float& value) { ESvaluelow_ = value; }
    float getESValueLow() const { return ESvaluelow_; }
    void  setESValueHigh(const float& value) { ESvaluehigh_ = value; }
    float getESValueHigh() const { return ESvaluehigh_; }
    void print(std::ostream& s) const {
      s << "ESMIPToGeVConstant: ES low/high " << ESvaluelow_ << " / " << ESvaluehigh_ <<" [GeV/MIP count]";
    }

  private:

    float ESvaluelow_;
    float ESvaluehigh_;
};

#endif
