#ifndef CondFormats_EcalObjects_EcalMGPAGainRatio_H
#define CondFormats_EcalObjects_EcalMGPAGainRatio_H
/**
 * Author: Shahram Rahatlou, University of Rome & INFN
 * Created: 22 Feb 2006
 * $Id: EcalMGPAGainRatio.h,v 1.3 2006/02/23 16:56:34 rahatlou Exp $
 **/


#include <iostream>

class EcalMGPAGainRatio {
  public:
    EcalMGPAGainRatio();
    EcalMGPAGainRatio(const EcalMGPAGainRatio & ratio);
    ~EcalMGPAGainRatio();

    float gain12Over6() const { return gain12Over6_; }
    float gain6Over1() const { return gain6Over1_; }

    void setGain12Over6(const float& g) { gain12Over6_ = g; }
    void setGain6Over1(const float& g)  { gain6Over1_ = g; }

    void print(std::ostream& s) const { s << "gain 12/6: " << gain12Over6_ << " gain 6/1: " << gain6Over1_; }

    EcalMGPAGainRatio& operator=(const EcalMGPAGainRatio& rhs);

  private:
    float gain12Over6_;
    float gain6Over1_;
};
#endif
