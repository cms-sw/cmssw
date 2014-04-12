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
  static const float innerRadius(PFGeometry::Layers_t layer)
    { return innerRadius_[layer]; }

  /// return outer radius of a given layer
  static const float outerRadius(PFGeometry::Layers_t layer)
    { return outerRadius_[layer]; }

  /// return inner position along z axis of a given layer
  static const float innerZ(PFGeometry::Layers_t layer)
    { return innerZ_[layer]; }

  /// return outer position along z axis of a given layer
  static const float outerZ(PFGeometry::Layers_t layer)
    { return outerZ_[layer]; }

  /// return cylinder used to propagate to barrel
  static const Cylinder& barrelBound(PFGeometry::Surface_t iSurf)
  { return *(cylinder_[unsigned(iSurf)]); }

  /// return disk used to propagate to negative endcap 
  static const Plane& negativeEndcapDisk(PFGeometry::Surface_t iSurf)
  { return *(negativeDisk_[unsigned(iSurf)]); }

  /// return disk used to propagate to positive endcap
  static const Plane& positiveEndcapDisk(PFGeometry::Surface_t iSurf)
  { return *(positiveDisk_[unsigned(iSurf)]); }

  /// return tan(theta) of the cylinder corner
  static float tanTh(PFGeometry::Surface_t iSurf)
  { return tanTh_[unsigned(iSurf)]; }

 private:
  static std::vector< float > innerRadius_;
  static std::vector< float > outerRadius_;
  static std::vector< float > innerZ_;
  static std::vector< float > outerZ_;

  static std::vector< ReferenceCountingPointer<Cylinder> > cylinder_;
  static std::vector< ReferenceCountingPointer<Plane> > negativeDisk_;
  static std::vector< ReferenceCountingPointer<Plane> > positiveDisk_;
  static std::vector< float > tanTh_;
};

#endif
