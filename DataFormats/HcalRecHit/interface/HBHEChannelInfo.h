#ifndef DataFormats_HcalRecHit_HBHEChannelInfo_h_
#define DataFormats_HcalRecHit_HBHEChannelInfo_h_

#include <cfloat>

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalRecHit/interface/HcalSpecialTimes.h"

/** \class HBHEChannelInfo
 *
 * Unpacked charge and TDC information in a format which works
 * for both QIE8 and QIE11
 */
class HBHEChannelInfo {
public:
  typedef HcalDetId key_type;

  static const unsigned MAXSAMPLES = 10;

  constexpr HBHEChannelInfo()
      : id_(),
        rawCharge_{0.},
        pedestal_{0.},
        pedestalWidth_{0.},
        gain_{0.},
        gainWidth_{0.},
        darkCurrent_{0},
        fcByPE_{0},
        lambda_{0},
        riseTime_{0.f},
        adc_{0},
        dFcPerADC_{0.f},
        recoShape_{0},
        nSamples_{0},
        soi_{0},
        capid_{0},
        hasTimeInfo_{false},
        hasEffectivePedestals_{false},
        dropped_{true},
        hasLinkError_{false},
        hasCapidError_{false} {}

  constexpr explicit HBHEChannelInfo(const bool hasTimeFromTDC, const bool hasEffectivePed)
      : id_(),
        rawCharge_{0.},
        pedestal_{0.},
        pedestalWidth_{0.},
        gain_{0.},
        gainWidth_{0.},
        darkCurrent_{0},
        fcByPE_{0},
        lambda_{0},
        riseTime_{0.f},
        adc_{0},
        dFcPerADC_{0.f},
        recoShape_{0},
        nSamples_{0},
        soi_{0},
        capid_{0},
        hasTimeInfo_(hasTimeFromTDC),
        hasEffectivePedestals_(hasEffectivePed),
        dropped_{true},
        hasLinkError_{false},
        hasCapidError_{false} {}

  constexpr void clear() {
    id_ = HcalDetId(0U);
    recoShape_ = 0;
    nSamples_ = 0;
    soi_ = 0;
    capid_ = 0;
    darkCurrent_ = 0;
    fcByPE_ = 0;
    lambda_ = 0, dropped_ = true;
    hasLinkError_ = false;
    hasCapidError_ = false;
  }

  constexpr void setChannelInfo(const HcalDetId& detId,
                                const int recoShape,
                                const unsigned nSamp,
                                const unsigned iSoi,
                                const int iCapid,
                                const double darkCurrent,
                                const double fcByPE,
                                const double lambda,
                                const bool linkError,
                                const bool capidError,
                                const bool dropThisChannel) {
    recoShape_ = recoShape;
    id_ = detId;
    nSamples_ = nSamp < MAXSAMPLES ? nSamp : MAXSAMPLES;
    soi_ = iSoi;
    capid_ = iCapid;
    darkCurrent_ = darkCurrent;
    fcByPE_ = fcByPE;
    lambda_ = lambda, dropped_ = dropThisChannel;
    hasLinkError_ = linkError;
    hasCapidError_ = capidError;
  }

  constexpr void tagAsDropped() { dropped_ = true; }

  // For speed, the "setSample" function does not perform bounds checking
  constexpr void setSample(const unsigned ts,
                           const uint8_t rawADC,
                           const float differentialChargeGain,
                           const double q,
                           const double ped,
                           const double pedWidth,
                           const double g,
                           const double gainWidth,
                           const float t) {
    rawCharge_[ts] = q;
    riseTime_[ts] = t;
    adc_[ts] = rawADC;
    dFcPerADC_[ts] = differentialChargeGain;
    pedestal_[ts] = ped;
    gain_[ts] = g;
    pedestalWidth_[ts] = pedWidth;
    gainWidth_[ts] = gainWidth;
  }

  // Inspectors
  constexpr HcalDetId id() const { return id_; }

  // access the recoShape
  constexpr int recoShape() const { return recoShape_; }

  constexpr unsigned nSamples() const { return nSamples_; }
  constexpr unsigned soi() const { return soi_; }
  constexpr int capid() const { return capid_; }
  constexpr bool hasTimeInfo() const { return hasTimeInfo_; }
  constexpr bool hasEffectivePedestals() const { return hasEffectivePedestals_; }
  constexpr double darkCurrent() const { return darkCurrent_; }
  constexpr double fcByPE() const { return fcByPE_; }
  constexpr double lambda() const { return lambda_; }
  constexpr bool isDropped() const { return dropped_; }
  constexpr bool hasLinkError() const { return hasLinkError_; }
  constexpr bool hasCapidError() const { return hasCapidError_; }

  // Direct read-only access to time slice arrays
  constexpr double const* rawCharge() const { return rawCharge_; }
  constexpr double const* pedestal() const { return pedestal_; }
  constexpr double const* pedestalWidth() const { return pedestalWidth_; }
  constexpr double const* gain() const { return gain_; }
  constexpr double const* gainWidth() const { return gainWidth_; }
  constexpr uint8_t const* adc() const { return adc_; }
  constexpr float const* dFcPerADC() const { return dFcPerADC_; }
  constexpr float const* riseTime() const {
    if (hasTimeInfo_)
      return riseTime_;
    else
      return nullptr;
  }

  // Indexed access to time slice quantities. No bounds checking.
  constexpr double tsRawCharge(const unsigned ts) const { return rawCharge_[ts]; }
  constexpr double tsPedestal(const unsigned ts) const { return pedestal_[ts]; }
  constexpr double tsPedestalWidth(const unsigned ts) const { return pedestalWidth_[ts]; }
  constexpr double tsGain(const unsigned ts) const { return gain_[ts]; }
  constexpr double tsGainWidth(const unsigned ts) const { return gainWidth_[ts]; }
  constexpr double tsCharge(const unsigned ts) const { return rawCharge_[ts] - pedestal_[ts]; }
  constexpr double tsEnergy(const unsigned ts) const { return (rawCharge_[ts] - pedestal_[ts]) * gain_[ts]; }
  constexpr uint8_t tsAdc(const unsigned ts) const { return adc_[ts]; }
  constexpr float tsDFcPerADC(const unsigned ts) const { return dFcPerADC_[ts]; }
  constexpr float tsRiseTime(const unsigned ts) const {
    return hasTimeInfo_ ? riseTime_[ts] : HcalSpecialTimes::UNKNOWN_T_NOTDC;
  }

  // Signal rise time measurement for the SOI, if available
  constexpr float soiRiseTime() const {
    return (hasTimeInfo_ && soi_ < nSamples_) ? riseTime_[soi_] : HcalSpecialTimes::UNKNOWN_T_NOTDC;
  }

  // The TS with the "end" index is not included in the window
  constexpr double chargeInWindow(const unsigned begin, const unsigned end) const {
    double sum = 0.0;
    const unsigned imax = end < nSamples_ ? end : nSamples_;
    for (unsigned i = begin; i < imax; ++i)
      sum += (rawCharge_[i] - pedestal_[i]);
    return sum;
  }

  constexpr double energyInWindow(const unsigned begin, const unsigned end) const {
    double sum = 0.0;
    const unsigned imax = end < nSamples_ ? end : nSamples_;
    for (unsigned i = begin; i < imax; ++i)
      sum += (rawCharge_[i] - pedestal_[i]) * gain_[i];
    return sum;
  }

  // The two following methods return MAXSAMPLES if the specified
  // window does not overlap with the samples stored
  constexpr unsigned peakChargeTS(const unsigned begin, const unsigned end) const {
    unsigned iPeak = MAXSAMPLES;
    double dmax = -DBL_MAX;
    const unsigned imax = end < nSamples_ ? end : nSamples_;
    for (unsigned i = begin; i < imax; ++i) {
      const double q = rawCharge_[i] - pedestal_[i];
      if (q > dmax) {
        dmax = q;
        iPeak = i;
      }
    }
    return iPeak;
  }

  constexpr unsigned peakEnergyTS(const unsigned begin, const unsigned end) const {
    unsigned iPeak = MAXSAMPLES;
    double dmax = -DBL_MAX;
    const unsigned imax = end < nSamples_ ? end : nSamples_;
    for (unsigned i = begin; i < imax; ++i) {
      const double e = (rawCharge_[i] - pedestal_[i]) * gain_[i];
      if (e > dmax) {
        dmax = e;
        iPeak = i;
      }
    }
    return iPeak;
  }

  // The following function can be used, for example,
  // in a check for presence of saturated ADC values
  constexpr uint8_t peakAdcValue(const unsigned begin, const unsigned end) const {
    uint8_t peak = 0;
    const unsigned imax = end < nSamples_ ? end : nSamples_;
    for (unsigned i = begin; i < imax; ++i)
      if (adc_[i] > peak)
        peak = adc_[i];
    return peak;
  }

private:
  HcalDetId id_;

  // Charge in fC for all time slices
  double rawCharge_[MAXSAMPLES];

  // Pedestal in fC
  double pedestal_[MAXSAMPLES];

  // Pedestal Width in fC
  double pedestalWidth_[MAXSAMPLES];

  // fC to GeV conversion factor
  double gain_[MAXSAMPLES];

  // fC to GeV conversion factor
  double gainWidth_[MAXSAMPLES];

  // needed for the dark current
  double darkCurrent_;
  double fcByPE_;
  double lambda_;

  // Signal rise time from TDC in ns (if provided)
  float riseTime_[MAXSAMPLES];

  // Raw QIE ADC values
  uint8_t adc_[MAXSAMPLES];

  // Differential fC/ADC gain. Needed for proper determination
  // of the ADC quantization error.
  float dFcPerADC_[MAXSAMPLES];

  // Reco Shapes
  int32_t recoShape_;

  // Number of time slices actually filled
  uint32_t nSamples_;

  // "Sample of interest" in the array of time slices
  uint32_t soi_;

  // QIE8 or QIE11 CAPID for the sample of interest
  int32_t capid_;

  // Flag indicating presence of the time info from TDC (QIE11)
  bool hasTimeInfo_;

  // Flag indicating use of effective pedestals
  bool hasEffectivePedestals_;

  // Flag indicating that this channel should be dropped
  // (typically, tagged bad from DB or zero-suppressed)
  bool dropped_;

  // Flags indicating presence of hardware errors
  bool hasLinkError_;
  bool hasCapidError_;
};

#endif  // DataFormats_HcalRecHit_HBHEChannelInfo_h_
