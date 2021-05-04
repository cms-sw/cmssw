//#define DO_THROW_UNINITIALIZED
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackingRecHit/interface/KfComponentsHolder.h"
#include "DataFormats/Math/interface/ProjectMatrix.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace {
#if defined(DO_THROW_UNINITIALIZED) || defined(DO_INTERNAL_CHECKS_BTR)
  inline void throwExceptionUninitialized(const char *where) {
    throw cms::Exception("BaseTrackerRecHit")
        << "Trying to access " << where
        << " for a RecHit that was read from disk, but since CMSSW_2_1_X local positions are transient.\n"
        << "If you want to get coarse position/error estimation from disk, please set: "
           "ComputeCoarseLocalPositionFromDisk = True \n "
        << " to the TransientTrackingRecHitBuilder you are using from "
           "RecoTracker/TransientTrackingRecHit/python/TTRHBuilders_cff.py";
  }
#endif
  void obsolete() { throw cms::Exception("BaseTrackerRecHit") << "CLHEP is obsolete for Tracker Hits"; }
}  // namespace

#if !defined(VI_DEBUG) && defined(DO_INTERNAL_CHECKS_BTR)
void BaseTrackerRecHit::check() const {
  if (!hasPositionAndError())
    throwExceptionUninitialized("localPosition or Error");
}
#endif

bool BaseTrackerRecHit::hasPositionAndError() const {
  //if det is present pos&err are available as well.
  //    //if det() is not present (null) the hit has been read from file and not updated
  return det();
}

void BaseTrackerRecHit::getKfComponents1D(KfComponentsHolder &holder) const {
#if defined(DO_THROW_UNINITIALIZED)
  if (!hasPositionAndError())
    throwExceptionUninitialized("getKfComponents");
#endif
  AlgebraicVector1 &pars = holder.params<1>();
  pars[0] = pos_.x();

  AlgebraicSymMatrix11 &errs = holder.errors<1>();
  errs(0, 0) = err_.xx();

  ProjectMatrix<double, 5, 1> &pf = holder.projFunc<1>();
  pf.index[0] = 3;

  holder.measuredParams<1>() = AlgebraicVector1(holder.tsosLocalParameters().At(3));
  holder.measuredErrors<1>() = holder.tsosLocalErrors().Sub<AlgebraicSymMatrix11>(3, 3);
}

void BaseTrackerRecHit::getKfComponents2D(KfComponentsHolder &holder) const {
#if defined(DO_THROW_UNINITIALIZED)
  if (!hasPositionAndError())
    throwExceptionUninitialized("getKfComponents");
#endif
  AlgebraicVector2 &pars = holder.params<2>();
  pars[0] = pos_.x();
  pars[1] = pos_.y();

  AlgebraicSymMatrix22 &errs = holder.errors<2>();
  errs(0, 0) = err_.xx();
  errs(0, 1) = err_.xy();
  errs(1, 1) = err_.yy();

  ProjectMatrix<double, 5, 2> &pf = holder.projFunc<2>();
  pf.index[0] = 3;
  pf.index[1] = 4;

  holder.measuredParams<2>() = AlgebraicVector2(&holder.tsosLocalParameters().At(3), 2);
  holder.measuredErrors<2>() = holder.tsosLocalErrors().Sub<AlgebraicSymMatrix22>(3, 3);
}

// obsolete (for what tracker is concerned...) interface
AlgebraicVector BaseTrackerRecHit::parameters() const {
  obsolete();
  return AlgebraicVector();
}

AlgebraicSymMatrix BaseTrackerRecHit::parametersError() const {
  obsolete();
  return AlgebraicSymMatrix();
}

AlgebraicMatrix BaseTrackerRecHit::projectionMatrix() const {
  obsolete();
  return AlgebraicMatrix();
}
