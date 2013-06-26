#ifndef JetMETCorrections_FFTJetObjects_L2ResScaleCalculator_h
#define JetMETCorrections_FFTJetObjects_L2ResScaleCalculator_h

#include <cassert>

#include "JetMETCorrections/FFTJetObjects/interface/AbsFFTSpecificScaleCalculator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

class L2ResScaleCalculator : public AbsFFTSpecificScaleCalculator
{
public:
    inline explicit L2ResScaleCalculator(const edm::ParameterSet& ps) 
        : m_radiusFactor(ps.getParameter<double>("radiusFactor")) {}

    inline virtual ~L2ResScaleCalculator() {}

    inline virtual void mapFFTJet(const reco::Jet& /* jet */,
                                  const reco::FFTJet<float>& fftJet,
                                  const math::XYZTLorentzVector& current,
                                  double* buf, const unsigned dim) const
    {
        if (dim != 2)
            throw cms::Exception("FFTJetBadConfig")
                << "In L2ResScaleCalculator::mapFFTJet: "
                << "invalid table dimensionality: "
                << dim << std::endl;
        assert(buf);
        const double radius = fftJet.f_recoScale();
        buf[0] = radius*m_radiusFactor;
        buf[1] = current.eta();
    }

private:
    double m_radiusFactor;
};

#endif // JetMETCorrections_FFTJetObjects_L2ResScaleCalculator_h
