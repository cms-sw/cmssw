#ifndef JetMETCorrections_FFTJetObjects_AbsFFTSpecificScaleCalculator_h
#define JetMETCorrections_FFTJetObjects_AbsFFTSpecificScaleCalculator_h

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/FFTJet.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "JetMETCorrections/FFTJetObjects/interface/FFTJetObjectFactory.h"

//
// A non-templated base class which can often be used to provide
// the mapping from jet quantities into lookup table variables
//
struct AbsFFTSpecificScaleCalculator
{
    inline virtual ~AbsFFTSpecificScaleCalculator() {}

    virtual void mapFFTJet(const reco::Jet& jet,
                           const reco::FFTJet<float>& fftJet,
                           const math::XYZTLorentzVector& current,
                           double* buf, unsigned dim) const = 0;
};

//
// The factory for classes derived from AbsFFTSpecificScaleCalculator.
// Note that there are no public constructors of this factory. All API
// is via the static wrapper.
//
class FFTSpecificScaleCalculatorFactory :
    public DefaultFFTJetObjectFactory<AbsFFTSpecificScaleCalculator>
{
    typedef DefaultFFTJetObjectFactory<AbsFFTSpecificScaleCalculator> Base;
    friend class StaticFFTJetObjectFactory<FFTSpecificScaleCalculatorFactory>;
    FFTSpecificScaleCalculatorFactory();
};

typedef StaticFFTJetObjectFactory<FFTSpecificScaleCalculatorFactory>
StaticFFTSpecificScaleCalculatorFactory;

AbsFFTSpecificScaleCalculator* parseFFTSpecificScaleCalculator(
    const edm::ParameterSet& ps, const std::string& tableDescription);

#endif // JetMETCorrections_FFTJetObjects_AbsFFTSpecificScaleCalculator_h
