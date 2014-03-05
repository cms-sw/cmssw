#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/Math/interface/ProjectMatrix.h"
#include "FWCore/Utilities/interface/Exception.h"



namespace {
  inline void
  throwExceptionUninitialized(const char *where)
  {
    throw cms::Exception("BaseTrackerRecHit") << 
      "Trying to access " << where << " for a RecHit that was read from disk, but since CMSSW_2_1_X local positions are transient.\n" <<
      "If you want to get coarse position/error estimation from disk, please set: ComputeCoarseLocalPositionFromDisk = True \n " <<
      " to the TransientTrackingRecHitBuilder you are using from RecoTracker/TransientTrackingRecHit/python/TTRHBuilders_cff.py";
  }
  
  void obsolete() {
    throw cms::Exception("BaseTrackerRecHit") << "CLHEP is obsolete for Tracker Hits";
  }
}

#ifdef EDM_LM_DEBUG
void BaseTrackerRecHit::check() const {
  if (!hasPositionAndError()) throwExceptionUninitialized("localPosition or Error");
}
#endif

bool BaseTrackerRecHit::hasPositionAndError() const {
  return det();
  
  //  return (err_.xx() != 0) || (err_.yy() != 0) || (err_.xy() != 0) ||
  //       (pos_.x()  != 0) || (pos_.y()  != 0) || (pos_.z()  != 0);
}



void
BaseTrackerRecHit::getKfComponents1D( KfComponentsHolder & holder ) const 
{
  // if (!hasPositionAndError()) throwExceptionUninitialized("getKfComponents");
  AlgebraicVector1 & pars = holder.params<1>();
  pars[0] = pos_.x(); 
  
  AlgebraicSymMatrix11 & errs = holder.errors<1>();
  errs(0,0) = err_.xx();
  
  AlgebraicMatrix15 & proj = holder.projection<1>();
  proj(0,3) = 1;
  
   holder.measuredParams<1>() = AlgebraicVector1( holder.tsosLocalParameters().At(3) );
   holder.measuredErrors<1>() = holder.tsosLocalErrors().Sub<AlgebraicSymMatrix11>( 3, 3 );
}

void
BaseTrackerRecHit::getKfComponents2D( KfComponentsHolder & holder ) const 
{
  //if (!hasPositionAndError()) throwExceptionUninitialized("getKfComponents");
   AlgebraicVector2 & pars = holder.params<2>();
   pars[0] = pos_.x(); 
   pars[1] = pos_.y();

   AlgebraicSymMatrix22 & errs = holder.errors<2>();
   errs(0,0) = err_.xx();
   errs(0,1) = err_.xy();
   errs(1,1) = err_.yy();

   
   AlgebraicMatrix25 & proj = holder.projection<2>();
   proj(0,3) = 1;
   proj(1,4) = 1;

   ProjectMatrix<double,5,2>  & pf = holder.projFunc<2>();
   pf.index[0] = 3;
   pf.index[1] = 4;
   holder.doUseProjFunc();

   holder.measuredParams<2>() = AlgebraicVector2( & holder.tsosLocalParameters().At(3), 2 );
   holder.measuredErrors<2>() = holder.tsosLocalErrors().Sub<AlgebraicSymMatrix22>( 3, 3 );

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
