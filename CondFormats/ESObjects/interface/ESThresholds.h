#ifndef CondFormats_ESObjects_ESThresholds_H
#define CondFormats_ESObjects_ESThresholds_H
#include <iostream>

class ESThresholds {

  public:

    ESThresholds();
    ESThresholds(const float & ts2, const float & zs);
    ~ESThresholds();

    void  setTS2Threshold(const float& value) { ts2_ = value; }
    float getTS2Threshold() const { return ts2_; }
    void  setZSThreshold(const float& value) { zs_ = value; }
    float getZSThreshold() const { return zs_; }

    void print(std::ostream& s) const {
      s << "ESThresholds: 2nd time sample / ZS threshold" << ts2_ << " / " << zs_ <<" [ADC count]";
    }

  private:

    float ts2_;
    float zs_;
};


#endif
