//#include "Utilities/Configuration/interface/Architecture.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/PerpendicularBoundPlaneBuilder.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include <iostream>

using namespace std;
int main() {
  PerpendicularBoundPlaneBuilder builder;

  GlobalVector v(1, 2, 3);

  BoundPlane* plane = builder(GlobalPoint(0, 0, 0), v);

  cout << "Local direction is " << plane->toLocal(v) << endl;
}
