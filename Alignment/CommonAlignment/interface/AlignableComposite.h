#ifndef Alignment_CommonAlignment_AlignableComposite_H
#define Alignment_CommonAlignment_AlignableComposite_H

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableSurface.h"

#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"
#include "Geometry/Vector/interface/Basic3DVector.h"

#include <vector>

class GeomDet;

/// Abstract base class for composites of Alignable objects.
/// The AlignableComposite is itself Alignable.
/// Apart from providing an interface to access components,
/// the AlignableComposite provides a convenient way of (mis)aligning
/// a droup of detectors as one rigid body.
/// The components of a Composite can themselves be composite,
/// providing a hierarchical view of the detector for alignment.
///
/// This is similar to the GeomDetUnit - GeomDet hierarchy, but the two
/// hierarchies are deliberately not coupled: The hierarchy for 
/// alignment is defined by the mechanical mounting of detectors
/// in various structures, while the GeomDet hierarchy is
/// optimised for pattern recognition.
///
/// Misalignment can be de-/reactivated (forwarded to components).
///

class AlignableComposite : public Alignable 
{

 public:

  AlignableComposite();

  explicit AlignableComposite( GeomDet* geomDet );

  virtual ~AlignableComposite() {}
  
  // The global position of the object 
  virtual GlobalPoint globalPosition() const { return theSurface.position(); }

  // The global orientation of the object 
  virtual RotationType globalRotation() const { return theSurface.rotation(); }

  // The global Position+Orientation (Surface) of the object 
  virtual const AlignableSurface& surface() const { return theSurface; }

  /// movement with respect to the GLOBAL CMS reference frame
  virtual void move( const GlobalVector& displacement ); 

  /// Movement of components with respect to the local composite 
  /// reference frame
  virtual void moveComponentsLocal( const LocalVector& localDisplacement ) 
  {
    this->move ( this->surface().toGlobal(localDisplacement) ) ;
  }

  /// Movement of a single component with respect to the local composite 
  /// reference frame
  virtual void moveComponentLocal( int i, 
				   const LocalVector& localDisplacement );

  /// Rotation intepreted such, that the orientation of the rotation
  /// axis is w.r.t. to the global coordinate system, however, this does NOT
  /// mean the center of the rotation. This is simply taken as the center of
  /// the Alignable-object 
  virtual void rotateInGlobalFrame( const RotationType& rotation );

  /// Set/add the AlignmentPositionError to all the components of the composite
  virtual void setAlignmentPositionError( const AlignmentPositionError& ape );
  virtual void addAlignmentPositionError( const AlignmentPositionError& ape );

  /// Adds the AlignmentPositionError (in x,y,z coordinates) that would result
  /// on the various components from a possible Rotation of a composite the 
  /// rotation matrix is in interpreted in GLOBAL coordinates
  virtual void addAlignmentPositionErrorFromRotation( const RotationType& rotation );

  /// Adds the AlignmentPositionError (in x,y,z coordinates) that would result
  /// on the various components from a possible Rotation of a composite the 
  /// rotation matrix is in interpreted in LOCAL  coordinates of the composite
  virtual void addAlignmentPositionErrorFromLocalRotation( const RotationType& rotation );

  /// Restore original position
  virtual void deactivateMisalignment ();

  /// Redo misalignment
  virtual void reactivateMisalignment ();

  /// Access to the GeomDet
  virtual GeomDet* geomDet() const { return theGeomDet; }


protected:
  void setSurface( const AlignableSurface& s) { theSurface = s; }
  /// Move Alignables in global frame without moving the associated Det
  virtual void moveAlignableOnly (const GlobalVector& displacement); 
  /// Rotate Alignables in global frame without rotating the associated Det
  virtual void rotateAlignableOnly (const RotationType& rotation);
  
protected:
  GeomDet* theGeomDet;

private:
  // Global position and orientation of the composite
  AlignableSurface  theSurface;


};

#endif




