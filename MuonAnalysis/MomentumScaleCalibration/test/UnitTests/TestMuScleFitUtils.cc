#include <iostream>

#include "MuonAnalysis/MomentumScaleCalibration/interface/MuScleFitUtils.h"

int main()
{
  std::cout << "Testing MuScleFitUtils" << std::endl;
  std::cout << std::endl;

  int iY = 0;
  double mass = 90.;
  double massResol = 1.;
  int iRes = 0;

  // Creating fake quantities
  for (int iy=0; iy<=MuScleFitUtils::nbins; ++iy) {
    MuScleFitUtils::GLZNorm[iY][iy] = 0.;
    for (int ix=0; ix<=MuScleFitUtils::nbins; ++ix) {
      MuScleFitUtils::GLZValue[iY][ix][iy] = 1.;
      MuScleFitUtils::GLZNorm[iY][iy] += MuScleFitUtils::GLZValue[iY][ix][iy];
    }
  }

  double prob = MuScleFitUtils::probability( mass, massResol,
                                             MuScleFitUtils::GLZValue, MuScleFitUtils::GLZNorm,
                                             iRes, iY );

  std::cout << "Probability = " << prob << std::endl;

}
