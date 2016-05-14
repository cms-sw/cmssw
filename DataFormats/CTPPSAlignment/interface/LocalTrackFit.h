/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#ifndef DataFormats_CTPPSAlignment_LocalTrackFit
#define DataFormats_CTPPSAlignment_LocalTrackFit

#include "TMath.h"

/**
 *\brief Local (linear) track description (or a fit result).
 * Uses global reference system.
 **/
struct LocalTrackFit
{
  /// the point where intercepts are measured, in mm
  double z0;

  /// slopes in rad
  double ax, ay;

  /// intercepts in mm
  double bx, by;

  /// the number of degrees of freedom
  signed int ndf;

  /// the residual sum of squares
  double chi_sq;

  LocalTrackFit(double _z0=0., double _ax=0., double _ay=0., double _bx=0., double _by=0., unsigned int _ndf=0, double _chi_sq=0.) : 
    z0(_z0), ax(_ax), ay(_ay), bx(_bx), by(_by), ndf(_ndf), chi_sq(_chi_sq) {}

  double PValue() const
    { return TMath::Prob(chi_sq, ndf); }
  
  double ChiSqPerNdf() const
    { return (ndf > 0) ? chi_sq / ndf : 0.; }

  void Eval(double z, double &x, double &y)
    {
      double ze = z - z0;
      x = ax*ze + bx;
      y = ay*ze + by;
    }
};

#endif

