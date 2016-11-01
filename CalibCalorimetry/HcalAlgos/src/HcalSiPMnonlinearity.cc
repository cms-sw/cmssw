#include <iostream>
#include "Math/Polynomial.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalSiPMnonlinearity.h"

// Assume parameters come to us from the reco side; i.e.,
// true pes = corfun(pixelsfired). But we want to invert that.
//
int HcalSiPMnonlinearity::getPixelsFired(int inpes) const
{
  ROOT::Math::Polynomial p(a2,b1,c0,-inpes);
  std::vector<double> roots = p.FindRealRoots();
  assert(roots.size());

  // all use cases tested over the full range of anticipated values;
  // the first root is always the right one.
  double realpix = roots[0];

  return realpix > 0 ? (int)(realpix+0.5) : 0;
}
