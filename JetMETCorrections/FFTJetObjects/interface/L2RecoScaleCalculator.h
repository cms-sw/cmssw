#ifndef JetMETCorrections_FFTJetObjects_L2RecoScaleCalculator_h
#define JetMETCorrections_FFTJetObjects_L2RecoScaleCalculator_h

#include <cassert>

#include "JetMETCorrections/FFTJetObjects/interface/AbsFFTSpecificScaleCalculator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

class L2RecoScaleCalculator : public AbsFFTSpecificScaleCalculator {
public:
  inline explicit L2RecoScaleCalculator(const edm::ParameterSet& ps)
      : m_radiusFactor(ps.getParameter<double>("radiusFactor")) {}

  inline ~L2RecoScaleCalculator() override {}

  inline void mapFFTJet(const reco::Jet& /* jet */,
                        const reco::FFTJet<float>& fftJet,
                        const math::XYZTLorentzVector& /* current */,
                        double* buf,
                        const unsigned dim) const override {
    if (dim != 1)
      throw cms::Exception("FFTJetBadConfig") << "In L2RecoScaleCalculator::mapFFTJet: "
                                              << "invalid table dimensionality: " << dim << std::endl;
    assert(buf);
    const double radius = fftJet.f_recoScale();
    buf[0] = radius * m_radiusFactor;
  }

private:
  double m_radiusFactor;
};

#endif  // JetMETCorrections_FFTJetObjects_L2RecoScaleCalculator_h
