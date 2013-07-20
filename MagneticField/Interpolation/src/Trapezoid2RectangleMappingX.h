#ifndef Trapezoid2RectangleMappingX_h
#define Trapezoid2RectangleMappingX_h

/** \class Trapezoid2RectangleMappingX
 *
 *  Maps a trapezoidal coordinate system into a cartesian one.
 *  It is assumed that x is the coordinate along the trapezoid bases while y is along
 *  the trapezoid height.
 *
 *  $Date: 2011/04/16 12:47:37 $
 *  $Revision: 1.6 $
 *  \author T. Todorov
 */


//#define DEBUG_GRID_TRM

#ifdef DEBUG_GRID_TRM
#include <iostream>
#endif

#include "FWCore/Utilities/interface/Visibility.h"

class dso_internal Trapezoid2RectangleMappingX {
public:

  Trapezoid2RectangleMappingX() {}

  /// normal trapezoid case, a != b
  Trapezoid2RectangleMappingX( double x0, double y0, double bovera, double h) :
    x0_(x0), y0_(y0), parallel_(false)
  {
    k_ = 2/h * (bovera-1.) / (bovera+1.);

#ifdef DEBUG_GRID_TRM
    std::cout << "Trapezoid2RectangleMappingX constructed with x0,y0 " << x0 << " " << y0 
 	 << " b/a= " << bovera << " h= " << h << std::endl;
#endif
  }

  /// special parallelogram case, a == b. The meaning of k changes.
  Trapezoid2RectangleMappingX( double x0, double y0, double k) :
    x0_(x0), y0_(y0), k_(k), parallel_(true)
  {
#ifdef DEBUG_GRID_TRM
    std::cout << "Trapezoid2RectangleMappingX constructed with x0,y0 " << x0 << " " << y0 
 	 << " k= " << k << std::endl;
#endif
  }

  void rectangle( double xtrap, double ytrap, 
		  double& xrec, double& yrec) const {

    yrec = ytrap - y0_;
    if (!parallel_) xrec = (xtrap - x0_) / (1.+ yrec*k_);
    else            xrec = xtrap - x0_ + k_*yrec;

#ifdef DEBUG_GRID
    std::cout << xtrap << " " << ytrap << " transformed to rectangle " << xrec << " " << yrec << std::endl;
#endif
  }

  void trapezoid( double xrec, double yrec, 
		  double& xtrap, double& ytrap) const {
    if (!parallel_) xtrap = x0_ + xrec * (1.+ yrec*k_);
    else            xtrap = x0_ + xrec - k_*yrec;
    ytrap = y0_ + yrec;

#ifdef DEBUG_GRID
    std::cout << xrec << " " << yrec << " transformed to trapezoid " << xtrap << " " << ytrap << std::endl;
#endif
  }

private:
  double x0_;
  double y0_;
  double k_;
  bool parallel_;
};

#endif
