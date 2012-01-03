// ClusterShapeHitFilter test
#define private public
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"
#undef private

#include <iostream>
int main() {

  ClusterShapeHitFilter filter(nullptr,nullptr,nullptr,nullptr);

  std::cout << "dump strip limits" << std::endl;
  for (int i=0; i!=StripKeys::N+1; i++) {
    std::cout << i << ": ";
    float const * p = filter.stripLimits[i].data[0];
    for (int j=0;j!=4; ++j)
      std::cout << p[j] << ", ";
    std::cout << std::endl;
  }

  std::cout << "\ndump pixel limits" << std::endl;
  for (int i=0; i!=PixelKeys::N+1; i++) {
     std::cout << i << ": ";
     float const * p = filter.pixelLimits[i].data[0][0];
     for (int j=0;j!=8; ++j)
       std::cout << p[j] << ", ";
     std::cout << std::endl;
  }

}
