#ifndef RecoLocalCalo_HcalRecAlgos_HcalPipelinePedestalAndGain_h_
#define RecoLocalCalo_HcalRecAlgos_HcalPipelinePedestalAndGain_h_

// HB/HE channel information stored for each pipeline "capacitor id"
class HcalPipelinePedestalAndGain {
public:
  inline HcalPipelinePedestalAndGain()
      : pedestal_(0.f), pedestalWidth_(0.f), effPedestal_(0.f), effPedestalWidth_(0.f), gain_(0.f), gainWidth_(0.f) {}

  inline HcalPipelinePedestalAndGain(const float i_pedestal,
                                     const float i_pedestalWidth,
                                     const float i_effPedestal,
                                     const float i_effPedestalWidth,
                                     const float i_gain,
                                     const float i_gainWidth)
      : pedestal_(i_pedestal),
        pedestalWidth_(i_pedestalWidth),
        effPedestal_(i_effPedestal),
        effPedestalWidth_(i_effPedestalWidth),
        gain_(i_gain),
        gainWidth_(i_gainWidth) {}

  inline float pedestal(const bool useEffectivePeds) const { return useEffectivePeds ? effPedestal_ : pedestal_; }

  inline float pedestalWidth(const bool useEffectivePeds) const {
    return useEffectivePeds ? effPedestalWidth_ : pedestalWidth_;
  }

  inline float gain() const { return gain_; }
  inline float gainWidth() const { return gainWidth_; }

private:
  float pedestal_;
  float pedestalWidth_;
  float effPedestal_;
  float effPedestalWidth_;
  float gain_;
  float gainWidth_;
};

#endif  // RecoLocalCalo_HcalRecAlgos_HcalPipelinePedestalAndGain_h_
