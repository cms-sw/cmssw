#ifndef Alignment_CommonAlignment_AlignableComposite_H
#define Alignment_CommonAlignment_AlignableComposite_H

#include "Alignment/CommonAlignment/interface/Alignable.h"

/// Abstract base class for composites of Alignable objects.
/// The AlignableComposite is itself Alignable.
/// Apart from providing an interface to access components,
/// the AlignableComposite provides a convenient way of (mis)aligning
/// a droup of detectors as one rigid body.
/// The components of a Composite can themselves be composite,
/// providing a hierarchical view of the detector for alignment.
///
/// This is similar to the GeomDetUnit - GeomDet hierarchy, but the two
/// hierarchies are deliberately not coupled: the hierarchy for 
/// alignment is defined by the mechanical mounting of detectors
/// in various structures, while the GeomDet hierarchy is
/// optimised for pattern recognition.
///
/// Misalignment can be de-/reactivated (forwarded to components).
///

class GeomDet;

class AlignableComposite : public Alignable 
{

public:

  /// Default constructor: surface position = 0, rotation = identity
  AlignableComposite();

  /// Constructor from GeomDet
  explicit AlignableComposite( const GeomDet* geomDet );

  virtual ~AlignableComposite() {}

  /// Provide all components, subcomponents etc. (cf. description in base class)
  virtual void recursiveComponents(std::vector<Alignable*> &result) const;
  
  /// Return the global position of the object 
  virtual const GlobalPoint& globalPosition() const { return theSurface.position(); }
  
  /// Return the global orientation of the object 
  virtual const RotationType& globalRotation() const { return theSurface.rotation(); }

  /// Return the Surface (global position and orientation) of the object 
  virtual const AlignableSurface& surface() const { return theSurface; }

  /// Move with respect to the global reference frame
  virtual void move( const GlobalVector& displacement ); 

  /// Move with respect to the local reference frame
  virtual void moveComponentsLocal( const LocalVector& localDisplacement );

  /// Move a single component with respect to the local reference frame
  virtual void moveComponentLocal( const int i, const LocalVector& localDisplacement );

  /// Rotation interpreted in global reference frame
  virtual void rotateInGlobalFrame( const RotationType& rotation );

  /// Rotation intepreted in the local reference frame
  virtual void rotateInLocalFrame( const RotationType& rotation );

  /// Set the AlignmentPositionError to all the components of the composite
  virtual void setAlignmentPositionError( const AlignmentPositionError& ape );

  /// Add the AlignmentPositionError to all the components of the composite
  virtual void addAlignmentPositionError( const AlignmentPositionError& ape );

  /// Add position error to all components as resulting from global rotation
  virtual void addAlignmentPositionErrorFromRotation( const RotationType& rotation );

  /// Add position error to all components as resulting from given local rotation
  virtual void addAlignmentPositionErrorFromLocalRotation( const RotationType& rotation );

  /// Restore original position
  virtual void deactivateMisalignment ();

  /// Restore misaligned position
  virtual void reactivateMisalignment ();

  /// Recursive printout of alignable structure
  virtual void dump() const;

  /// Return alignment data
  virtual Alignments* alignments() const;

  /// Return vector of alignment errors
  virtual AlignmentErrors* alignmentErrors() const;

protected:
  void setSurface( const AlignableSurface& s) { theSurface = s; }
  
private:
  AlignableSurface  theSurface;   // Global position and orientation of the composite

};

#endif
