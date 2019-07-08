#include "DataFormats/TrackingRecHit/interface/KfComponentsHolder.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <typeinfo>

template <unsigned int D>
void KfComponentsHolder::genericFill_(const TrackingRecHit &hit) {
  typedef typename AlgebraicROOTObject<D, 5>::Matrix MatD5;
  typedef typename AlgebraicROOTObject<D, D>::SymMatrix SMatDD;
  typedef typename AlgebraicROOTObject<D>::Vector VecD;

  params<D>() = asSVector<D>(hit.parameters());
  errors<D>() = asSMatrix<D>(hit.parametersError());

  MatD5 &&H = asSMatrix<D, 5>(hit.projectionMatrix());
  projFunc<D>().fromH(H);

  measuredParams<D>() = H * (*tsosLocalParameters_);
  measuredErrors<D>() = ROOT::Math::Similarity(H, (*tsosLocalErrors_));
}

void KfComponentsHolder::genericFill(const TrackingRecHit &hit) {
  switch (hit.dimension()) {
    case 1:
      genericFill_<1>(hit);
      break;
    case 2:
      genericFill_<2>(hit);
      break;
    case 3:
      genericFill_<3>(hit);
      break;
    case 4:
      genericFill_<4>(hit);
      break;
    case 5:
      genericFill_<5>(hit);
      break;
    default:
      throw cms::Exception("Rec hit of invalid dimension (not 1,2,3,4,5)")
          << "The dimension was " << hit.dimension() << ", type is " << typeid(hit).name() << "\n";
  }
}
