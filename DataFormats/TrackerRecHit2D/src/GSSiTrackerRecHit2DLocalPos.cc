#include "DataFormats/TrackerRecHit2D/interface/GSSiTrackerRecHit2DLocalPos.h"

void 
GSSiTrackerRecHit2DLocalPos::getKfComponents( KfComponentsHolder & holder ) const 
{
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
