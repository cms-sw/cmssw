#ifndef Alignment_CommonAlignment_AlignableComposite_H
#define Alignment_CommonAlignment_AlignableComposite_H

#include "Alignment/CommonAlignment/interface/Alignable.h"

/// Abstract base class for composites of Alignable objects.
/// The AlignableComposite is itself Alignable.
/// Apart from providing an interface to access components,
/// the AlignableComposite provides a convenient way of (mis)aligning
/// a group of detectors as one rigid body.
/// The components of a Composite can themselves be composite,
/// providing a hierarchical view of the detector for alignment.
///
/// This is similar to the GeomDetUnit - GeomDet hierarchy, but the two
/// hierarchies are deliberately not coupled: the hierarchy for 
/// alignment is defined by the mechanical mounting of detectors
/// in various structures, while the GeomDet hierarchy is
/// optimised for pattern recognition.
///
/// Note that AlignableComposite owns (and deletes in its destructor)
/// all its component which are added by addComponent. 

class GeomDet;

class AlignableComposite : public Alignable 
{

public:
  /// Constructor for a composite with given rotation.
  /// Position is found from average of daughters' positions later.
  /// Default values for backward compatibility with MuonAlignment.
  AlignableComposite( align::ID = 0,
		      StructureType = align::invalid,
		      const RotationType& = RotationType() );

  /// deleting its components
  virtual ~AlignableComposite();

  /// Add a component and set its mother to this alignable.
  /// (Note: The component will be adopted, e.g. later deleted.)
  /// Also find average position of this composite from its sensors' positions.
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
  virtual StructureType alignableObjectId() const { return theStructureType; }

  /// Recursive printout of alignable structure
  virtual void dump() const;

  /// Return alignment data
  virtual Alignments* alignments() const;

  /// Return vector of alignment errors
  virtual AlignmentErrors* alignmentErrors() const;

protected:
  /// Constructor from GeomDet, only for use in AlignableDet
  explicit AlignableComposite( const GeomDet* geomDet );

  void setSurface( const AlignableSurface& s) { theSurface = s; }
  
  StructureType theStructureType;

private:

  Alignables theComponents; // direct daughters
};

#endif
