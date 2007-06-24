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

  /// Constructor from GeomDet
  explicit AlignableComposite( const GeomDet* geomDet );

  /// Constructor for a composite with given rotation.
  /// Position is found from average of daughters' positions later.
  /// Default arguments for backward compatibility with empty constructor.
  AlignableComposite( const DetId& = DetId(),
		      AlignableObjectIdType = AlignableObjectId::invalid,
		      const RotationType& = RotationType() );

  virtual ~AlignableComposite();

  /// Add a component and set its mother to this alignable.
  /// Also find average position of this alignable.
  virtual void addComponent( Alignable* component );

  /// Return vector of direct components
  virtual Alignables components() const { return theComponents; }

  /// Provide all components, subcomponents etc. (cf. description in base class)
  virtual void recursiveComponents(Alignables &result) const;

  /// Move with respect to the global reference frame
  virtual void move( const GlobalVector& displacement ); 

  /// Move with respect to the local reference frame
  virtual void moveComponentsLocal( const LocalVector& localDisplacement );

  /// Move a single component with respect to the local reference frame
  virtual void moveComponentLocal( const int i, const LocalVector& localDisplacement );

  /// Rotation interpreted in global reference frame
  virtual void rotateInGlobalFrame( const RotationType& rotation );

  /// Set the AlignmentPositionError to all the components of the composite
  virtual void setAlignmentPositionError( const AlignmentPositionError& ape );

  /// Add the AlignmentPositionError to all the components of the composite
  virtual void addAlignmentPositionError( const AlignmentPositionError& ape );

  /// Add position error to all components as resulting from global rotation
  virtual void addAlignmentPositionErrorFromRotation( const RotationType& rotation );

  /// Add position error to all components as resulting from given local rotation
  virtual void addAlignmentPositionErrorFromLocalRotation( const RotationType& rotation );

  /// Return the alignable type identifier
  virtual int alignableObjectId() const { return static_cast<int>(theStructureType); }

  /// Recursive printout of alignable structure
  virtual void dump() const;

  /// Return alignment data
  virtual Alignments* alignments() const;

  /// Return vector of alignment errors
  virtual AlignmentErrors* alignmentErrors() const;

protected:
  void setSurface( const AlignableSurface& s) { theSurface = s; }
  
private:

  AlignableObjectIdType theStructureType;

  Alignables theComponents; // direct daughters
};

#endif
