#include "MagneticField/Interpolation/src/Grid3D.h"
#include <iostream>

void Grid3D::dump() const
{
  for (int j=0; j<gridb().nodes(); ++j) {
    for (int k=0; k<gridc().nodes(); ++k) {
      for (int i=0; i<grida().nodes(); ++i) {
        std::cout << grida().node(i) << " " << gridb().node(j) << " " << gridc().node(k) << " " << operator()(i,j,k) << std::endl;
      }
    }
  }
}
