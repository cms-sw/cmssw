#include "HFEGammaSLCorrector.h"

#include <cmath>

namespace hf_egamma {

  double eSeLCorrected(double es, double el, double pc, double px, double py) {
    double x = std::log(el / 100);
    double y = es / el;
    return pc + px * x + py * y;
  }

  double eSeLCorrected(double es, double el, int era) {
    double pc = 0.0, px = 0.0, py = 0.0;

    switch (era) {
      case (0):  //Data 41
        pc = -1.02388e-1;
        px = -1.51130e-1;
        py = 9.88514e-1;
        break;
      case (1):  //Fall 10 MC
        pc = -4.06012e-2;
        px = -1.34769e-1;
        py = 9.90877e-1;
        break;
      case (2):  //Spring 11 MC
        pc = 5.98732e-3;
        px = -1.74767e-1;
        py = 9.84610e-1;
        break;
      case (3):  //Summer 11 MC
        pc = -0.036416;
        px = -0.195854;
        py = 0.980633;
        break;
      case (4):  //July 5 Data
        pc = -0.008077;
        px = -0.216002;
        py = 0.976393;
        break;
    }

    //After fitting the 2D histogram, we find a y-intercept b, a slope m,
    //and a point x0 around which we choose to rotate the data points. We
    //will map (x,y) --> (x,y') where y' = pc + px*x + py*y, with
    //pc = sin(atan(m))*x0 - cos(atan(m))*(m*x0+b), px = -sin(atan(m)),
    //and py = cos(atan(m)). This transformation preserves the x-value of the
    //data point and takes y' to be the y-value of the original point after it
    //is rotated through angle atan(m) (to flatten the line of best fit) and
    //transposed vertically downward by b (to make the line of best fit coincide with the x-axis).

    return eSeLCorrected(es, el, pc, px, py);
  }
}  // namespace hf_egamma
