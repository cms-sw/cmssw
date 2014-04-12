#include <cassert>

#include "JetMETCorrections/FFTJetObjects/interface/AbsFFTJetAdjuster.h"
#include "FWCore/Utilities/interface/Exception.h"

template<class MyJet, class Adjustable>
struct FFTSimpleScalingAdjuster : public AbsFFTJetAdjuster<MyJet, Adjustable>
{
    inline virtual ~FFTSimpleScalingAdjuster() {}

    virtual void adjust(const MyJet& /* jet */, const Adjustable& in,
                        const double* factors, const unsigned lenFactors,
                        Adjustable* out) const
    {
        if (lenFactors != 1U)
            throw cms::Exception("FFTJetBadConfig")
                << "In FFTSimpleScalingAdjuster::adjust: wrong number of "
                << "scales (expected 1, got " << lenFactors << ")\n";
        assert(factors);
        assert(out);
        *out = in;
        *out *= factors[0];
    }
};

template<class MyJet, class Adjustable>
struct FFTUncertaintyAdjuster : public AbsFFTJetAdjuster<MyJet, Adjustable>
{
    inline virtual ~FFTUncertaintyAdjuster() {}

    virtual void adjust(const MyJet& /* jet */, const Adjustable& in,
                        const double* factors, const unsigned lenFactors,
                        Adjustable* out) const
    {
        if (lenFactors != 1U)
            throw cms::Exception("FFTJetBadConfig")
                << "In FFTUncertaintyAdjuster::adjust: wrong number of "
                << "scales (expected 1, got " << lenFactors << ")\n";
        assert(factors);
        assert(out);
        *out = in;
        const double s = factors[0];
        out->setVariance(in.variance() + s*s);
    }
};

template<class MyJet, class Adjustable>
struct FFTScalingAdjusterWithUncertainty : 
    public AbsFFTJetAdjuster<MyJet, Adjustable>
{
    inline virtual ~FFTScalingAdjusterWithUncertainty() {}

    virtual void adjust(const MyJet& /* jet */, const Adjustable& in,
                        const double* factors, const unsigned lenFactors,
                        Adjustable* out) const
    {
        if (lenFactors != 2U)
            throw cms::Exception("FFTJetBadConfig")
                << "In FFTScalingAdjusterWithUncertainty::adjust: wrong "
                << "number of scales (expected 2, got " << lenFactors << ")\n";
        assert(factors);
        assert(out);
        *out = in;
        *out *= factors[0];
        const double s = factors[1];
        out->setVariance(in.variance() + s*s);
    }
};
