#ifndef DataFormats_HcalRecHit_HBHEChannelInfo_h_
#define DataFormats_HcalRecHit_HBHEChannelInfo_h_

#include <array>
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
        : rawCharge_{}, pedestal_{}, gain_{}, adc_{},
          hasTimeInfo_(false) {clear();}

    inline explicit HBHEChannelInfo(const bool hasTimeFromTDC)
        : rawCharge_{}, pedestal_{}, gain_{}, adc_{},
          hasTimeInfo_(hasTimeFromTDC) {clear();}

    inline void clear()
    {
        id_ = HcalDetId(0U);
        nSamples_ = 0;
        soi_ = 0;
        capid_ = 0;
        dropped_ = true;
        hasLinkError_ = false;
        hasCapidError_ = false;
        for (unsigned i=0; i<MAXSAMPLES; ++i)
            riseTime_[i] = HcalSpecialTimes::UNKNOWN_T_NOTDC;
    }

    inline void setChannelInfo(const HcalDetId& detId, const unsigned nSamp,
                               const unsigned iSoi, const int iCapid,
                               const bool linkError, const bool capidError,
                               const bool dropThisChannel)
    {
        id_ = detId;
        nSamples_ = nSamp < MAXSAMPLES ? nSamp : MAXSAMPLES;
        soi_ = iSoi;
        capid_ = iCapid;
        dropped_ = dropThisChannel;
        hasLinkError_ = linkError;
        hasCapidError_ = capidError;
    }

    inline void tagAsDropped()
        {dropped_ = true;}

    // For speed, the "setSample" function does not perform bounds checking
    inline void setSample(const unsigned ts, const uint8_t rawADC,
                          const double q, const double ped,
                          const double g, const float t)
    {
        rawCharge_[ts] = q;
        pedestal_[ts] = ped;
        gain_[ts] = g;
        adc_[ts] = rawADC;
        riseTime_[ts] = hasTimeInfo_ ? t : HcalSpecialTimes::UNKNOWN_T_NOTDC;
    }

    // Inspectors
    inline HcalDetId id() const {return id_;}

    inline unsigned nSamples() const {return nSamples_;}
    inline unsigned soi() const {return soi_;}
    inline int capid() const {return capid_;}
    inline bool hasTimeInfo() const {return hasTimeInfo_;}
    inline bool isDropped() const {return dropped_;}
    inline bool hasLinkError() const {return hasLinkError_;}
    inline bool hasCapidError() const {return hasCapidError_;}

    // Direct read-only access to time slice arrays.
    // Note that only first "nSamples()" elements of
    // these arrays will have meaningful values.
    inline const std::array<double,MAXSAMPLES>& rawCharge() const {return rawCharge_;}
    inline const std::array<double,MAXSAMPLES>& pedestal() const {return pedestal_;}
    inline const std::array<double,MAXSAMPLES>& gain() const {return gain_;}
    inline const std::array<uint8_t,MAXSAMPLES>& adc() const {return adc_;}
    inline const std::array<float,MAXSAMPLES>& riseTime() const {return riseTime_;}

    // Indexed access to time slice quantities. No bounds checking.
    inline double tsRawCharge(const unsigned ts) const {return rawCharge_[ts];}
    inline double tsPedestal(const unsigned ts) const {return pedestal_[ts];}
    inline double tsCharge(const unsigned ts) const
        {return rawCharge_[ts] - pedestal_[ts];}
    inline double tsEnergy(const unsigned ts) const
        {return (rawCharge_[ts] - pedestal_[ts])*gain_[ts];}
    inline double tsGain(const unsigned ts) const {return gain_[ts];}
    inline uint8_t tsAdc(const unsigned ts) const {return adc_[ts];}
    inline float tsRiseTime(const unsigned ts) const {return riseTime_[ts];}

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
    std::array<double,MAXSAMPLES> rawCharge_;

    // Pedestal in fC
    std::array<double,MAXSAMPLES> pedestal_;

    // fC to GeV conversion factor (can depend on CAPID)
    std::array<double,MAXSAMPLES> gain_;

    // Signal rise time from TDC in ns (if provided)
    std::array<float,MAXSAMPLES> riseTime_;

    // Raw QIE ADC values
    std::array<uint8_t,MAXSAMPLES> adc_;

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
