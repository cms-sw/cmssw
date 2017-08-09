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

  BowedSurfaceDeformation* clone() const override;

  /// specific type, i.e. SurfaceDeformationFactory::kBowedSurface
  int type() const override;

  /// correction to add to local position depending on 
  /// - track parameters in local frame (from LocalTrajectoryParameters):
  ///   * track position as Local2DPoint(x,y)
  ///   * track angles   as LocalTrackAngles(dxdz, dydz)
  /// - length of surface (local y-coordinate)
  /// - width of surface (local x-coordinate)
  Local2DVector positionCorrection(const Local2DPoint &localPos,
                                           const LocalTrackAngles &localAngles,
                                           double length, double width) const override;

  /// update information with parameters of 'other',
  /// false in case the type or some parameters do not match and thus
  /// the information cannot be used (then no changes are done),
  /// true if merge was successful
  bool add(const SurfaceDeformation &other) override;
  
  /// parameters, i.e. sagittae as given in the constructor
  std::vector<double> parameters() const override;

  // the size
  static constexpr unsigned int parSize = 3;
  static constexpr unsigned int parameterSize() { return parSize; }

  /// minimum size of vector that is accepted by constructor from vector
  static constexpr unsigned int minParameterSize() { return parameterSize(); }
  /// maximum size of vector that is accepted by constructor from vector
  static constexpr unsigned int maxParameterSize() { return parameterSize(); }

 private:
  double theSagittaX;
  double theSagittaY;
  double theSagittaXY;
  // double theRelWidthLowY; // could be used for non-rectangular modules
};

#endif
