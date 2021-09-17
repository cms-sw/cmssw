#ifndef JetMETCorrections_FFTJetObjects_FFTJetScaleCalculators_h
#define JetMETCorrections_FFTJetObjects_FFTJetScaleCalculators_h

#include <cassert>
#include <cmath>

#include "JetMETCorrections/FFTJetObjects/interface/AbsFFTSpecificScaleCalculator.h"
#include "JetMETCorrections/FFTJetObjects/interface/AbsFFTJetScaleCalculator.h"
#include "FWCore/Utilities/interface/Exception.h"

template <class MyJet, class Adjustable>
class FFTEtaLogPtConeRadiusMapper : public AbsFFTJetScaleCalculator<MyJet, Adjustable> {
public:
  inline explicit FFTEtaLogPtConeRadiusMapper(std::shared_ptr<npstat::AbsMultivariateFunctor> f)
      : AbsFFTJetScaleCalculator<MyJet, Adjustable>(f) {}

private:
  inline void map(const MyJet& jet, const Adjustable& current, double* buf, const unsigned dim) const override {
    assert(buf);
    if (dim != 3)
      throw cms::Exception("FFTJetBadConfig") << "In FFTEtaLogPtConeRadiusMapper::map: "
                                              << "invalid table dimensionality: " << dim << std::endl;
    buf[0] = current.vec().eta();
    buf[1] = log(current.vec().pt());
    buf[2] = jet.getFFTSpecific().f_recoScale();
  }
};

template <class MyJet, class Adjustable>
class FFTSpecificScaleCalculator : public AbsFFTJetScaleCalculator<MyJet, Adjustable> {
public:
  //
  // This class will assume the ownership of the
  // AbsFFTSpecificScaleCalculator object provided
  // in the constructor
  //
  inline FFTSpecificScaleCalculator(std::shared_ptr<npstat::AbsMultivariateFunctor> f,
                                    const AbsFFTSpecificScaleCalculator* p)
      : AbsFFTJetScaleCalculator<MyJet, Adjustable>(f), calc_(p) {
    assert(p);
  }

  inline ~FFTSpecificScaleCalculator() override { delete calc_; }

private:
  inline void map(const MyJet& jet, const Adjustable& current, double* buf, const unsigned dim) const override {
    return calc_->mapFFTJet(jet, jet.getFFTSpecific(), current.vec(), buf, dim);
  }

  const AbsFFTSpecificScaleCalculator* calc_;
};

#endif  // JetMETCorrections_FFTJetObjects_FFTJetScaleCalculators_h
