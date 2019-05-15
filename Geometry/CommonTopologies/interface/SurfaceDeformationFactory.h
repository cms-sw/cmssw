#ifndef Geometry_CommonTopologies_SurfaceDeformationFactory_H
#define Geometry_CommonTopologies_SurfaceDeformationFactory_H

/// SurfaceDeformationFactory
///
/// Factory for concrete implementations of SurfaceDeformation
///
///  \author    : Gero Flucke
///  date       : October 2010

#include <vector>
#include <string>

class SurfaceDeformation;

namespace SurfaceDeformationFactory {
  enum Type {
    // rigid body has no deformations! kRigidBody = 0,
    kBowedSurface = 1,  // BowedSurfaceDeformation
    kTwoBowedSurfaces,  // TwoBowedSurfacesDeformation
    kNoDeformations     // To please compilers
  };

  /// convert string to 'Type' - exception if string is not known
  Type surfaceDeformationType(const std::string &typeString);
  std::string surfaceDeformationTypeName(const Type &type);

  /// Create an instance of the concrete implementations of
  /// the 'SurfaceDeformation' interface
  /// First argument 'type' must match one of the enums defined above
  /// and the size of 'params' must match the expectation of the
  /// concrete type (exception otherwise).
  SurfaceDeformation *create(int type, const std::vector<double> &params);
  SurfaceDeformation *create(const std::vector<double> &params);

}  // namespace SurfaceDeformationFactory

#endif
