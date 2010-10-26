#ifndef Geometry_CommonTopologies_SurfaceDeformation_H
#define Geometry_CommonTopologies_SurfaceDeformation_H

/// SurfaceDeformation
///
/// Abstract base class for corrections to be applied to 
/// 2D local positions on a surface if the surface is not perfectly
/// following its parameterisaton (e.g. bows for a Plane).
///
///  \author    : Gero Flucke
///  date       : October 2010
///  $Revision$
///  $Date$
///  (last update by $Author$)

#include "DataFormats/GeometryVector/interface/Vector2DBase.h"
#include "DataFormats/GeometryVector/interface/LocalTag.h"
// class AlgebraicVector5; doesn't work, it's a typedef...
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

#include <vector>

class SurfaceDeformation
{
 public:
  typedef Vector2DBase<double, LocalTag> Local2DVector;

  virtual SurfaceDeformation* clone() const = 0;

  /// specific type, i.e. SurfaceDeformationFactory::Type
  virtual int type() const = 0;

  /// correction to add to local position depending on 
  /// - track paramters, i.e.
  ///    trackPred[0] q/p : 
  ///    trackPred[1] dxdz: direction tangent in local xz-plane
  ///    trackPred[2] dydz: direction tangent in local yz-plane
  ///    trackPred[3] x   : local x-coordinate
  ///    trackPred[4] y   : local y-coordinate
  /// - length of surface (local y-coordinate)
  /// - width of surface (local x-coordinate)
  virtual Local2DVector positionCorrection(const AlgebraicVector5 &trackPred,
					   double length, double width) const = 0;

  /// update information with parameters of 'other',
  /// false in case the type or some parameters do not match and thus
  /// the information cannot be used (then no changes are done),
  /// true if merge was successful
  virtual bool add(const SurfaceDeformation &other) = 0;
  
  // Seems like only GeometryAligner and derived classes need access
  // to parameters, so we could make it a friend and protect parameters()...
  // friend class GeometryAligner; // to be able to call parameters
  // protected:

  /// parameters - interpretation left to the concrete implementation
  virtual std::vector<double> parameters() const = 0;

};

#endif
