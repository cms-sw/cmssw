#include "RecoPixelVertexing/PixelLowPtUtilities/src/LowPtClusterShapeSeedComparitor.cc"
#include <iostream>

void go(GlobalPoint const * g) {
  GlobalVector v[3];
  bool ok = getGlobalDirs(g,v);
  for(int i=0; i!=3; i++)
    std::cout << g[i] << std::endl;
  if (!ok) std::cout << "failed"  << std::endl;
  else 
  for(int i=0; i!=3; i++)
    std::cout << v[i] << std::endl;
  std::cout << "norm" << v.mag2() << std::endl;

}


int main() {

  GlobalPoint g1[3] = {
    GlobalPoint(1., 3., 0.),
    GlobalPoint(-1., 1., 1.),
    GlobalPoint(1., -1., 2.)
  };

  GlobalPoint g2[3] = {
    GlobalPoint(1., -1., 0.),
    GlobalPoint(-1., 1., 1.),
    GlobalPoint(1., -3., 2.)
  };

  GlobalPoint g3[3] = {
    GlobalPoint(1., 3., 0.),
    GlobalPoint(1., 1., -1.),
    GlobalPoint(1., -1., -2.)
  };

  go(g1);
  go(g2);
  go(g3);

  return 0;
}
