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
class HBHEChannelInfo
{
public:
    typedef HcalDetId key_type;

    static const unsigned MAXSAMPLES = 10;

    inline HBHEChannelInfo()
      : rawCharge_{0.}, pedestal_{0.}, pedestalWidth_{0.},
        gain_{0.}, gainWidth_{0.}, riseTime_{0.f}, adc_{0},
        dFcPerADC_{0.f}, hasTimeInfo_(false) {clear();}

    inline explicit HBHEChannelInfo(const bool hasTimeFromTDC)
      : rawCharge_{0.}, pedestal_{0.}, pedestalWidth_{0.},
        gain_{0.}, gainWidth_{0.}, riseTime_{0.f}, adc_{0},
        dFcPerADC_{0.f}, hasTimeInfo_(hasTimeFromTDC) {clear();}

    inline void clear()
    {
        id_ = HcalDetId(0U);
        recoShape_ = 0;
        nSamples_ = 0;
        soi_ = 0;
        capid_ = 0;
        darkCurrent_ = 0;
        fcByPE_ = 0;
        lambda_ = 0,
        dropped_ = true;
        hasLinkError_ = false;
        hasCapidError_ = false;
    }

    inline void setChannelInfo(const HcalDetId& detId, const int recoShape, const unsigned nSamp,
                               const unsigned iSoi, const int iCapid,
                               const double darkCurrent, const double fcByPE, const double lambda,
                               const bool linkError, const bool capidError,
                               const bool dropThisChannel)
    {
        recoShape_ = recoShape;
        id_ = detId;
        nSamples_ = nSamp < MAXSAMPLES ? nSamp : MAXSAMPLES;
        soi_ = iSoi;
        capid_ = iCapid;
        darkCurrent_ = darkCurrent;
        fcByPE_ = fcByPE;
        lambda_ = lambda,
        dropped_ = dropThisChannel;
        hasLinkError_ = linkError;
        hasCapidError_ = capidError;
    }

    inline void tagAsDropped()
        {dropped_ = true;}

    // For speed, the "setSample" function does not perform bounds checking
    inline void setSample(const unsigned ts, const uint8_t rawADC,
                          const float differentialChargeGain, const double q,
                          const double ped, const double pedWidth,
                          const double g, const double gainWidth,
                          const float t)
    {
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
    inline HcalDetId id() const {return id_;}

    // access the recoShape
    inline int recoShape() const { return recoShape_;}

    inline unsigned nSamples() const {return nSamples_;}
    inline unsigned soi() const {return soi_;}
    inline int capid() const {return capid_;}
    inline bool hasTimeInfo() const {return hasTimeInfo_;}
    inline double darkCurrent() const {return darkCurrent_;}
    inline double fcByPE() const {return fcByPE_;}
    inline double lambda() const {return lambda_;}
    inline bool isDropped() const {return dropped_;}
    inline bool hasLinkError() const {return hasLinkError_;}
    inline bool hasCapidError() const {return hasCapidError_;}

    // Direct read-only access to time slice arrays
    inline const double* rawCharge() const {return rawCharge_;}
    inline const double* pedestal() const {return pedestal_;}
    inline const double* pedestalWidth() const {return pedestalWidth_;}
    inline const double* gain() const {return gain_;}
    inline const double* gainWidth() const {return gainWidth_;}
    inline const uint8_t* adc() const {return adc_;}
    inline const float* dFcPerADC() const {return dFcPerADC_;}
    inline const float* riseTime() const
        {if (hasTimeInfo_) return riseTime_; else return nullptr;}

    // Indexed access to time slice quantities. No bounds checking.
    inline double tsRawCharge(const unsigned ts) const {return rawCharge_[ts];}
    inline double tsPedestal(const unsigned ts) const {return pedestal_[ts];}
    inline double tsPedestalWidth(const unsigned ts) const {return pedestalWidth_[ts];}
    inline double tsGain(const unsigned ts) const {return gain_[ts];}
    inline double tsGainWidth(const unsigned ts) const {return gainWidth_[ts];}
    inline double tsCharge(const unsigned ts) const
        {return rawCharge_[ts] - pedestal_[ts];}
    inline double tsEnergy(const unsigned ts) const
        {return (rawCharge_[ts] - pedestal_[ts])*gain_[ts];}
    inline uint8_t tsAdc(const unsigned ts) const {return adc_[ts];}
    inline float tsDFcPerADC(const unsigned ts) const {return dFcPerADC_[ts];}
    inline float tsRiseTime(const unsigned ts) const
        {return hasTimeInfo_ ? riseTime_[ts] : HcalSpecialTimes::UNKNOWN_T_NOTDC;}

    // Signal rise time measurement for the SOI, if available
    inline float soiRiseTime() const
    {
        return (hasTimeInfo_ && soi_ < nSamples_) ?
                riseTime_[soi_] : HcalSpecialTimes::UNKNOWN_T_NOTDC;
    }

    // The TS with the "end" index is not included in the window
    inline double chargeInWindow(const unsigned begin, const unsigned end) const
    {
        double sum = 0.0;
        const unsigned imax = end < nSamples_ ? end : nSamples_;
        for (unsigned i=begin; i<imax; ++i)
            sum += (rawCharge_[i] - pedestal_[i]);
        return sum;
    }

    inline double energyInWindow(const unsigned begin, const unsigned end) const
    {
        double sum = 0.0;
        const unsigned imax = end < nSamples_ ? end : nSamples_;
        for (unsigned i=begin; i<imax; ++i)
            sum += (rawCharge_[i] - pedestal_[i])*gain_[i];
        return sum;
    }

    // The two following methods return MAXSAMPLES if the specified
    // window does not overlap with the samples stored
    inline unsigned peakChargeTS(const unsigned begin, const unsigned end) const
    {
        unsigned iPeak = MAXSAMPLES;
        double dmax = -DBL_MAX;
        const unsigned imax = end < nSamples_ ? end : nSamples_;
        for (unsigned i=begin; i<imax; ++i)
        {
            const double q = rawCharge_[i] - pedestal_[i];
            if (q > dmax)
            {
                dmax = q;
                iPeak = i;
            }
        }
        return iPeak;
    }

    inline unsigned peakEnergyTS(const unsigned begin, const unsigned end) const
    {
        unsigned iPeak = MAXSAMPLES;
        double dmax = -DBL_MAX;
        const unsigned imax = end < nSamples_ ? end : nSamples_;
        for (unsigned i=begin; i<imax; ++i)
        {
            const double e = (rawCharge_[i] - pedestal_[i])*gain_[i];
            if (e > dmax)
            {
                dmax = e;
                iPeak = i;
            }
        }
        return iPeak;
    }

    // The following function can be used, for example,
    // in a check for presence of saturated ADC values
    inline uint8_t peakAdcValue(const unsigned begin, const unsigned end) const
    {
        uint8_t peak = 0;
        const unsigned imax = end < nSamples_ ? end : nSamples_;
        for (unsigned i=begin; i<imax; ++i)
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

    // Flag indicating that this channel should be dropped
    // (typically, tagged bad from DB or zero-suppressed)
    bool dropped_;

    // Flags indicating presence of hardware errors
    bool hasLinkError_;
    bool hasCapidError_;
};

#endif // DataFormats_HcalRecHit_HBHEChannelInfo_h_
