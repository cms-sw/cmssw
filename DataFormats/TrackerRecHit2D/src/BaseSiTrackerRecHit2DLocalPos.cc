#include "DataFormats/TrackerRecHit2D/interface/BaseSiTrackerRecHit2DLocalPos.h"
#include "FWCore/Utilities/interface/Exception.h"

bool BaseSiTrackerRecHit2DLocalPos::hasPositionAndError() const {
    return (err_.xx() != 0) || (err_.yy() != 0) || (err_.xy() != 0) ||
           (pos_.x()  != 0) || (pos_.y()  != 0) || (pos_.z()  != 0);
}

LocalPoint BaseSiTrackerRecHit2DLocalPos::localPosition() const {
    if (!hasPositionAndError()) throw cms::Exception("BaseSiTrackerRecHit2DLocalPos") << 
        "Trying to access the localPosition of a RecHit that was read from disk, but since CMSSW_2_1_X localPosition is transient.\n";
    return pos_;
}

LocalError BaseSiTrackerRecHit2DLocalPos::localPositionError() const{ 
    if (!hasPositionAndError()) throw cms::Exception("BaseSiTrackerRecHit2DLocalPos") << 
        "Trying to access the localPositionError of a RecHit that was read from disk, but since CMSSW_2_1_X localPositionError is transient.\n";
    return err_;
}


void 
BaseSiTrackerRecHit2DLocalPos::getKfComponents( KfComponentsHolder & holder ) const 
{
   if (!hasPositionAndError()) throw cms::Exception("BaseSiTrackerRecHit2DLocalPos") << 
        "Trying to access the KfComponents of a RecHit that was read from disk, but since CMSSW_2_1_X local positions are transient.\n";
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

   holder.measuredParams<2>() = AlgebraicVector2( & holder.tsosLocalParameters().At(3), 2 );
   holder.measuredErrors<2>() = holder.tsosLocalErrors().Sub<AlgebraicSymMatrix22>( 3, 3 );

   //std::cout << "======== MYSELF ==========" << std::endl;
   //holder.dump<2>();
   //std::cout << "======== GENERIC ==========" << std::endl;
   //holder.genericFill(*this);
   //holder.dump<2>();
}
