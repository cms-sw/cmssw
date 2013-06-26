#ifndef JetMETCorrections_FFTJetObjects_AbsFFTJetAdjuster_h
#define JetMETCorrections_FFTJetObjects_AbsFFTJetAdjuster_h

template<class Jet, class Adjustable>
struct AbsFFTJetAdjuster
{
    typedef Jet jet_type;
    typedef Adjustable adjustable_type;

    inline virtual ~AbsFFTJetAdjuster() {}

    virtual void adjust(const Jet& jet, const Adjustable& in,
                        const double* factors, unsigned lenFactors,
                        Adjustable* out) const = 0;
};

#endif // JetMETCorrections_FFTJetObjects_AbsFFTJetAdjuster_h
