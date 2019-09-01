#ifndef GeomVector_jacobians_
#define GeomVector_jacobians_

#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

AlgebraicMatrix65 jacobianCurvilinearToCartesian(const GlobalVector& momentum, int charge);
AlgebraicMatrix56 jacobianCartesianToCurvilinear(const GlobalVector& momentum, int charge);

#endif
