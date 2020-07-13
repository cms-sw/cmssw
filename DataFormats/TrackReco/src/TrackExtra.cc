#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/fillCovariance.h"

using namespace reco;

TrackExtra::TrackExtra(const Point &outerPosition,
                       const Vector &outerMomentum,
                       bool ok,
                       const Point &innerPosition,
                       const Vector &innerMomentum,
                       bool iok,
                       const CovarianceMatrix &outerCov,
                       unsigned int outerId,
                       const CovarianceMatrix &innerCov,
                       unsigned int innerId,
                       PropagationDirection seedDir,
                       edm::RefToBase<TrajectorySeed> seedRef)
    :

      TrackExtraBase(),
      outerPosition_(outerPosition),
      outerMomentum_(outerMomentum),
      outerOk_(ok),
      outerDetId_(outerId),
      innerPosition_(innerPosition),
      innerMomentum_(innerMomentum),
      innerOk_(iok),
      innerDetId_(innerId),
      seedDir_(seedDir),
      seedRef_(seedRef) {
  index idx = 0;
  for (index i = 0; i < dimension; ++i) {
    for (index j = 0; j <= i; ++j) {
      outerCovariance_[idx] = outerCov(i, j);
      innerCovariance_[idx] = innerCov(i, j);
      ++idx;
    }
  }
}

void TrackExtra::clearOuter() {
  outerPosition_ = Point();
  std::fill(outerCovariance_, outerCovariance_ + covarianceSize, 0.);
  outerOk_ = false;
}

void TrackExtra::clearInner() {
  innerPosition_ = Point();
  std::fill(innerCovariance_, innerCovariance_ + covarianceSize, 0.);
  innerOk_ = false;
}

TrackExtra::CovarianceMatrix TrackExtra::outerStateCovariance() const {
  CovarianceMatrix v;
  fillCovariance(v, outerCovariance_);
  return v;
}

TrackExtra::CovarianceMatrix TrackExtra::innerStateCovariance() const {
  CovarianceMatrix v;
  fillCovariance(v, innerCovariance_);
  return v;
}

TrackExtra::CovarianceMatrix &TrackExtra::fillOuter(CovarianceMatrix &v) const {
  return fillCovariance(v, outerCovariance_);
}

TrackExtra::CovarianceMatrix &TrackExtra::fillInner(CovarianceMatrix &v) const {
  return fillCovariance(v, innerCovariance_);
}
