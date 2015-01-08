#ifndef CondFormats_HcalObjects_InterpolatedPulse_h_
#define CondFormats_HcalObjects_InterpolatedPulse_h_

#include <cassert>
#include "FWCore/Utilities/interface/Exception.h"

#include "boost/serialization/access.hpp"
#include "boost/serialization/version.hpp"

template<unsigned MaxLen>
class InterpolatedPulse
{
    template <unsigned Len2> friend class InterpolatedPulse;

public:
    // Would normally do "static const" but genreflex has problems for it
    enum {maxlen = MaxLen};

    // Default constructor creates a pulse which is zero everywhere
    inline InterpolatedPulse()
        : tmin_(0.0), width_(1.0), length_(2U)
    {
        zeroOut();
    }

    // Constructor from a single integer creates a pulse with the given
    // number of discrete steps which is zero everywhere
    inline explicit InterpolatedPulse(const unsigned len)
        : tmin_(0.0), width_(1.0), length_(len)
    {
        if (length_ < 2 || length_ > MaxLen) throw cms::Exception(
            "In InterpolatedPulse constructor: invalid length");
        zeroOut();
    }

    inline InterpolatedPulse(const double tmin, const double tmax,
                             const unsigned len)
        : tmin_(tmin), width_(tmax - tmin), length_(len)
    {
        if (length_ < 2 || length_ > MaxLen) throw cms::Exception(
            "In InterpolatedPulse constructor: invalid length");
        if (width_ <= 0.0) throw cms::Exception(
            "In InterpolatedPulse constructor: invalid pulse width");
        zeroOut();
    }

    template <typename Real>
    inline InterpolatedPulse(const double tmin, const double tmax,
                             const Real* values, const unsigned len)
        : tmin_(tmin), width_(tmax - tmin)
    {
        if (width_ <= 0.0) throw cms::Exception(
            "In InterpolatedPulse constructor: invalid pulse width");
        setShape(values, len);
    }

    // Efficient copy constructor. Do not copy undefined values.
    inline InterpolatedPulse(const InterpolatedPulse& r)
        : tmin_(r.tmin_), width_(r.width_), length_(r.length_)
    {
        double* buf = &pulse_[0];
        const double* rbuf = &r.pulse_[0];
        for (unsigned i=0; i<length_; ++i)
            *buf++ = *rbuf++;
    }

    // Converting copy constructor
    template <unsigned Len2>
    inline InterpolatedPulse(const InterpolatedPulse<Len2>& r)
        : tmin_(r.tmin_), width_(r.width_), length_(r.length_)
    {
        if (length_ > MaxLen) throw cms::Exception(
            "In InterpolatedPulse copy constructor: buffer is not long enough");
        double* buf = &pulse_[0];
        const double* rbuf = &r.pulse_[0];
        for (unsigned i=0; i<length_; ++i)
            *buf++ = *rbuf++;
    }

    // Efficient assignment operator
    inline InterpolatedPulse& operator=(const InterpolatedPulse& r)
    {
        if (this != &r)
        {
            tmin_ = r.tmin_;
            width_ = r.width_;
            length_ = r.length_;
            double* buf = &pulse_[0];
            const double* rbuf = &r.pulse_[0];
            for (unsigned i=0; i<length_; ++i)
                *buf++ = *rbuf++;
        }
        return *this;
    }

    // Converting assignment operator
    template <unsigned Len2>
    inline InterpolatedPulse& operator=(const InterpolatedPulse<Len2>& r)
    {
        if (r.length_ > MaxLen) throw cms::Exception(
            "In InterpolatedPulse::operator=: buffer is not long enough");
        tmin_ = r.tmin_;
        width_ = r.width_;
        length_ = r.length_;
        double* buf = &pulse_[0];
        const double* rbuf = &r.pulse_[0];
        for (unsigned i=0; i<length_; ++i)
            *buf++ = *rbuf++;
        return *this;
    }

    template <typename Real>
    inline void setShape(const Real* values, const unsigned len)
    {
        if (len < 2 || len > MaxLen) throw cms::Exception(
            "In InterpolatedPulse::setShape: invalid length");
        assert(values);
        length_ = len;
        double* buf = &pulse_[0];
        for (unsigned i=0; i<len; ++i)
            *buf++ = *values++;
    }

    // Zero out the signal
    inline void zeroOut()
    {
        for (unsigned i=0; i<length_; ++i)
            pulse_[i] = 0.0;
    }

    // Simple inspectors
    inline const double* getPulse() const {return &pulse_[0];}
    inline unsigned getLength() const {return length_;}
    inline double getStartTime() const {return tmin_;}
    inline double getStopTime() const {return tmin_ + width_;}
    inline double getPulseWidth() const {return width_;}
    inline double getTimeStep() const {return width_/(length_ - 1U);}

    // Simple modifiers
    inline void setStartTime(const double newStartTime)
        {tmin_ = newStartTime;}

    inline void setPulseWidth(const double newWidth)
    {
        if (newWidth <= 0.0) throw cms::Exception(
            "In InterpolatedPulse::setPulseWidth: invalid pulse width");
        width_ = newWidth;
    }

    // Get the pulse value at the given time using linear interpolation
    inline double operator()(const double t) const
    {
        const volatile double tmax = tmin_ + width_;
        if (t < tmin_ || t > tmax)
            return 0.0;
        const unsigned lm1 = length_ - 1U;
        const double step = width_/lm1;
        const double nSteps = (t - tmin_)/step;
        unsigned nbelow = nSteps;
        unsigned nabove = nbelow + 1;
        if (nabove > lm1)
        {
            nabove = lm1;
            nbelow = nabove - 1U;
        }
        const double delta = nSteps - nbelow;
        return pulse_[nbelow]*(1.0 - delta) + pulse_[nabove]*delta;
    }

    inline double derivative(const double t) const
    {
        const volatile double tmax = tmin_ + width_;
        if (t < tmin_ || t > tmax)
            return 0.0;
        const unsigned lm1 = length_ - 1U;
        const double step = width_/lm1;
        const double nSteps = (t - tmin_)/step;
        unsigned nbelow = nSteps;
        unsigned nabove = nbelow + 1;
        if (nabove > lm1)
        {
            nabove = lm1;
            nbelow = nabove - 1U;
        }
        const double delta = nSteps - nbelow;
        if ((nbelow == 0U && delta <= 0.5) || (nabove == lm1 && delta >= 0.5))
            return (pulse_[nabove] - pulse_[nbelow])/step;
        else if (delta >= 0.5)
        {
            const double lower = pulse_[nabove] - pulse_[nbelow];
            const double upper = pulse_[nabove+1U] - pulse_[nabove];
            return (upper*(delta - 0.5) + lower*(1.5 - delta))/step;
        }
        else
        {
            const double lower = pulse_[nbelow] - pulse_[nbelow-1U];
            const double upper = pulse_[nabove] - pulse_[nbelow];
            return (lower*(0.5 - delta) + upper*(0.5 + delta))/step;
        }
    }

    inline double secondDerivative(const double t) const
    {
        const volatile double tmax = tmin_ + width_;
        if (t < tmin_ || t > tmax || length_ < 3U)
            return 0.0;
        const unsigned lm1 = length_ - 1U;
        const double step = width_/lm1;
        const double stepSq = step*step;
        const double nSteps = (t - tmin_)/step;
        unsigned nbelow = nSteps;
        unsigned nabove = nbelow + 1;
        if (nabove > lm1)
        {
            nabove = lm1;
            nbelow = nabove - 1U;
        }

        if (nbelow == 0U)
        {
            // The first interval
            return (pulse_[2] - 2.0*pulse_[1] + pulse_[0])/stepSq;
        }
        else if (nabove == lm1)
        {
            // The last interval
            return (pulse_[lm1] - 2.0*pulse_[lm1-1U] + pulse_[lm1-2U])/stepSq;
        }
        else
        {
            // One of the middle intervals
            const double lower = pulse_[nbelow-1U] - 2.0*pulse_[nbelow] + pulse_[nabove];
            const double upper = pulse_[nbelow] - 2.0*pulse_[nabove] + pulse_[nabove+1U];
            const double delta = nSteps - nbelow;
            return (lower*(1.0 - delta) + upper*delta)/stepSq;
        }
    }

    inline InterpolatedPulse& operator*=(const double scale)
    {
        if (scale != 1.0)
        {
            double* buf = &pulse_[0];
            for (unsigned i=0; i<length_; ++i)
                *buf++ *= scale;
        }
        return *this;
    }

    // Add another pulse to this one. Note that addition of another pulse
    // will not change the start time or the width of _this_ pulse. The
    // added pulse will be truncated as needed.
    template <unsigned Len2>
    inline InterpolatedPulse& operator+=(const InterpolatedPulse<Len2>& r)
    {
        const double step = width_/(length_ - 1U);
        for (unsigned i=0; i<length_; ++i)
            pulse_[i] += r(tmin_ + i*step);
        return *this;
    }

    template <unsigned Len2>
    inline bool operator==(const InterpolatedPulse<Len2>& r) const
    {
        if (!(tmin_ == r.tmin_ && width_ == r.width_ && length_ == r.length_))
            return false;
        const double* buf = &pulse_[0];
        const double* rbuf = &r.pulse_[0];
        for (unsigned i=0; i<length_; ++i)
            if (*buf++ != *rbuf++)
                return false;
        return true;
    }

    template <unsigned Len2>
    inline bool operator!=(const InterpolatedPulse<Len2>& r) const
        {return !(*this == r);}

    // Simple trapezoidal integration
    inline double getIntegral() const
    {
        const double* buf = &pulse_[0];
        long double sum = buf[0]/2.0;
        const unsigned nIntervals = length_ - 1U;
        for (unsigned i=1U; i<nIntervals; ++i)
            sum += buf[i];
        sum += buf[nIntervals]/2.0;
        return sum*width_/nIntervals;
    }

    inline void setIntegral(const double newValue)
    {
        const double integ = this->getIntegral();
        if (integ == 0.0) throw cms::Exception(
            "In InterpolatedPulse::setIntegral division by zero");
        *this *= (newValue/integ);
    }

    inline double getPeakValue() const
    {
        const double* buf = &pulse_[0];
        double peak = buf[0];
        for (unsigned i=1U; i<length_; ++i)
            if (buf[i] > peak)
                peak = buf[i];
        return peak;
    }

    inline void setPeakValue(const double newValue)
    {
        const double peak = this->getPeakValue();
        if (peak == 0.0) throw cms::Exception(
            "In InterpolatedPulse::setPeakValue: division by zero");
        *this *= (newValue/peak);
    }

private:
    double pulse_[MaxLen];
    double tmin_;
    double width_;
    unsigned length_;

    friend class boost::serialization::access;

    template<class Archive>
    inline void serialize(Archive & ar, unsigned /* version */)
    {
        ar & tmin_ & width_ & length_;

        // In case we are reading, it may be useful to verify
        // that the length is reasonable
        if (length_ > MaxLen) throw cms::Exception(
            "In InterpolatedPulse::serialize: buffer is not long enough");

        for (unsigned i=0; i<length_; ++i)
            ar & pulse_[i];
    }
};

// boost serialization version number for this template
namespace boost {
    namespace serialization {
        template<unsigned MaxLen>
        struct version<InterpolatedPulse<MaxLen> >
        {
            BOOST_STATIC_CONSTANT(int, value = 1);
        };
    }
}

#endif // CondFormats_HcalObjects_InterpolatedPulse_h_
