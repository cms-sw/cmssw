#ifndef RecoParticleFlow_PFTracking_PFGeometry_h
#define RecoParticleFlow_PFTracking_PFGeometry_h 

#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"

// system include files
#include <vector>

/**\class PFGeometry 
\brief General CMS geometry parameters used during Particle Flow 
reconstruction or drawing.
All methods and members are static

\author Renaud Bruneliere
\date   August 2006
\todo   Move this class out of here, or modify it so that it does not use the Geometry/Surface package anymore.
*/

class Cylinder;
class Disk;
class Plane;

class PFGeometry {
 public:
  typedef enum {
    BeamPipe = 0,
    PS1 = 1,
    PS2 = 2,
    ECALBarrel = 3,
    ECALEndcap = 4,
    HCALBarrel = 5,
    HCALEndcap = 6,
    HOBarrel = 7,
    NPoints = 8
  } Layers_t;

  typedef enum {
    BeamPipeWall = 0,
    PS1Wall = 1,
    PS2Wall = 2,
    ECALInnerWall = 3,
    HCALInnerWall = 4,
    HCALOuterWall = 5,
    HOInnerWall = 6,
    HOOuterWall = 7,
    NSurfPoints = 8
  } Surface_t;

  /// constructor
  PFGeometry();

  /// destructor
  virtual ~PFGeometry() { }

  /// return inner radius of a given layer
  float innerRadius(PFGeometry::Layers_t layer) const
    { return innerRadius_[layer]; }

  /// return outer radius of a given layer
  float outerRadius(PFGeometry::Layers_t layer) const
    { return outerRadius_[layer]; }

  /// return inner position along z axis of a given layer
  float innerZ(PFGeometry::Layers_t layer) const
    { return innerZ_[layer]; }

  /// return outer position along z axis of a given layer
  float outerZ(PFGeometry::Layers_t layer) const
    { return outerZ_[layer]; }

  /// return tan(theta) of the cylinder corner
  float tanTh(PFGeometry::Surface_t iSurf) const
  { return tanTh_[unsigned(iSurf)]; }

 private:
  std::vector< float > innerRadius_;
  std::vector< float > outerRadius_;
  std::vector< float > innerZ_;
  std::vector< float > outerZ_;
  std::vector< float > tanTh_;
};

#endif
