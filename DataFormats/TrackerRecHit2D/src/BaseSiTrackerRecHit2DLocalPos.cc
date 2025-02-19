#include "DataFormats/TrackerRecHit2D/interface/BaseSiTrackerRecHit2DLocalPos.h"
#include "DataFormats/Math/interface/ProjectMatrix.h"
#include "FWCore/Utilities/interface/Exception.h"

bool BaseSiTrackerRecHit2DLocalPos::hasPositionAndError() const {
    return (err_.xx() != 0) || (err_.yy() != 0) || (err_.xy() != 0) ||
           (pos_.x()  != 0) || (pos_.y()  != 0) || (pos_.z()  != 0);
}

LocalPoint BaseSiTrackerRecHit2DLocalPos::localPosition() const {
    if (!hasPositionAndError()) throwExceptionUninitialized("localPosition");
    return pos_;
}

LocalError BaseSiTrackerRecHit2DLocalPos::localPositionError() const{ 
    if (!hasPositionAndError()) throwExceptionUninitialized("localPositionError");
    return err_;
}


void 
BaseSiTrackerRecHit2DLocalPos::getKfComponents( KfComponentsHolder & holder ) const 
{
   if (!hasPositionAndError()) throwExceptionUninitialized("getKfComponents");
   //std::cout << "Call to KfComponentsHolder::genericFill should be optimized here " << std::endl;
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

   //std::cout << "======== MYSELF ==========" << std::endl;
   //holder.dump<2>();
   //std::cout << "======== GENERIC ==========" << std::endl;
   //holder.genericFill(*this);
   //holder.dump<2>();
}

void
BaseSiTrackerRecHit2DLocalPos::throwExceptionUninitialized(const char *where) const
{
   throw cms::Exception("BaseSiTrackerRecHit2DLocalPos") << 
     "Trying to access " << where << " for a RecHit that was read from disk, but since CMSSW_2_1_X local positions are transient.\n" <<
     "If you want to get coarse position/error estimation from disk, please set: ComputeCoarseLocalPositionFromDisk = True \n " <<
     " to the TransientTrackingRecHitBuilder you are using from RecoTracker/TransientTrackingRecHit/python/TTRHBuilders_cff.py";
}

