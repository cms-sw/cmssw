#ifndef JetMETCorrections_FFTJetObjects_AbsFFTJetScaleCalculator_h
#define JetMETCorrections_FFTJetObjects_AbsFFTJetScaleCalculator_h

#include "JetMETCorrections/InterpolationTables/interface/AbsMultivariateFunctor.h"

#include <memory>
#include <vector>

template <class Jet, class Adjustable>
class AbsFFTJetScaleCalculator {
public:
  typedef Jet jet_type;
  typedef Adjustable adjustable_type;

  inline explicit AbsFFTJetScaleCalculator(std::shared_ptr<npstat::AbsMultivariateFunctor> f)
      : functor(f), buffer_(f->minDim()) {}

  inline virtual ~AbsFFTJetScaleCalculator() {}

  inline double scale(const Jet& jet, const Adjustable& current) const {
    const unsigned dim = buffer_.size();
    double* buf = dim ? &buffer_[0] : static_cast<double*>(nullptr);
    this->map(jet, current, buf, dim);
    return (*functor)(buf, dim);
  }

private:
  AbsFFTJetScaleCalculator() = delete;

  virtual void map(const Jet& jet, const Adjustable& current, double* buf, unsigned dim) const = 0;

  std::shared_ptr<npstat::AbsMultivariateFunctor> functor;
  mutable std::vector<double> buffer_;
};

#endif  // JetMETCorrections_FFTJetObjects_AbsFFTJetScaleCalculator_h
