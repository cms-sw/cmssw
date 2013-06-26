#ifndef Geometry_CommonTopologies_BowedSurfaceDeformation_H
#define Geometry_CommonTopologies_BowedSurfaceDeformation_H

/// BowedSurfaceAlignmentParameters
///
/// Class to apply corrections to local positions resulting
/// from a non-planar surface. The bows are parametrised using
/// Legendre polynomials up to second order, excluding 
/// 0th and 1st order that are already treated by local w
/// shift and rotations around local u and v axes.
///
///  \author    : Gero Flucke
///  date       : October 2010
///  $Revision: 1.3 $
///  $Date: 2012/01/26 10:56:43 $
///  (last update by $Author: innocent $)

#include "Geometry/CommonTopologies/interface/SurfaceDeformation.h"

// already included in the above:
// #include <vector>

class BowedSurfaceDeformation : public SurfaceDeformation
{
 public:
  /// constructor from sagittae, i.e. coefficients of Legendre polynomials
  BowedSurfaceDeformation(double sagittaX, double sagittaXY, double sagittaY) :
    theSagittaX(sagittaX),theSagittaY(sagittaY),  theSagittaXY(sagittaXY)  {  }
  /// constructor from vector of sagittae, parameters.size() must be
  /// between minParameterSize() and maxParameterSize()
  BowedSurfaceDeformation(const std::vector<double> &parameters);

  virtual BowedSurfaceDeformation* clone() const;

  /// specific type, i.e. SurfaceDeformationFactory::kBowedSurface
  virtual int type() const;

  /// correction to add to local position depending on 
  /// - track parameters in local frame (from LocalTrajectoryParameters):
  ///   * track position as Local2DPoint(x,y)
  ///   * track angles   as LocalTrackAngles(dxdz, dydz)
  /// - length of surface (local y-coordinate)
  /// - width of surface (local x-coordinate)
  virtual Local2DVector positionCorrection(const Local2DPoint &localPos,
                                           const LocalTrackAngles &localAngles,
                                           double length, double width) const;

  /// update information with parameters of 'other',
  /// false in case the type or some parameters do not match and thus
  /// the information cannot be used (then no changes are done),
  /// true if merge was successful
  virtual bool add(const SurfaceDeformation &other);
  
  /// parameters, i.e. sagittae as given in the constructor
  virtual std::vector<double> parameters() const;

  /// minimum size of vector that is accepted by constructor from vector
  static unsigned int minParameterSize() { return 3;}
  /// maximum size of vector that is accepted by constructor from vector
  static unsigned int maxParameterSize() { return 3;}

 private:
  double theSagittaX;
  double theSagittaY;
  double theSagittaXY;
  // double theRelWidthLowY; // could be used for non-rectangular modules
};

#endif
