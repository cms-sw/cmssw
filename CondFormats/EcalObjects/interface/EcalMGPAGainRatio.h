#ifndef CondFormats_EcalObjects_EcalMGPAGainRatio_H
#define CondFormats_EcalObjects_EcalMGPAGainRatio_H

class EcalMGPAGainRatio {
  public:
    EcalMGPAGainRatio();
    EcalMGPAGainRatio(const EcalMGPAGainRatio & ratio);
    ~EcalMGPAGainRatio();

    float gain12Over6() const { return gain12Over6_; }
    float gain6Over1() const { return gain6Over1_; }

    void setGain12Over6(const float& g) { gain12Over6_ = g; }
    void setGain6Over1(const float& g)  { gain6Over1_ = g; }

  private:
    float gain12Over6_;
    float gain6Over1_;
};
#endif
