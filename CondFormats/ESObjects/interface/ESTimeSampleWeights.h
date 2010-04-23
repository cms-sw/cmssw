#ifndef CondFormats_ESObjects_ESTimeSampleWeights_H
#define CondFormats_ESObjects_ESTimeSampleWeights_H
#include <iostream>

class ESTimeSampleWeights {

  public:

    ESTimeSampleWeights();
    ESTimeSampleWeights(const float & w0, const float & w1, const float & w2);
    ~ESTimeSampleWeights();

    void  setWeightForTS0(const float& value) { w0_ = value; }
    float getWeightForTS0() const { return w0_; }
    void  setWeightForTS1(const float& value) { w1_ = value; }
    float getWeightForTS1() const { return w1_; }
    void  setWeightForTS2(const float& value) { w2_ = value; }
    float getWeightForTS2() const { return w2_; }

    void print(std::ostream& s) const {
      s<<"ESTimeSampleWeights: "<<w0_<<" "<<w1_<<" "<<w2_;
    }

  private:

    float w0_; 
    float w1_;
    float w2_;
};


#endif
