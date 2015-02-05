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
  /// Position can be found from average of daughters' positions later,
  /// using addComponent(Alignable*).
  AlignableComposite( align::ID id,
		      StructureType aType,
		      const RotationType& rot = RotationType() );

  /// deleting its components
  virtual ~AlignableComposite();

  /// Add a component and set its mother to this alignable.
  /// (Note: The component will be adopted, e.g. later deleted.)
  /// Also find average position of this composite from its modules' positions.
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

  /// Set the AlignmentPositionError (if this Alignable is a Det) and,
  /// if (propagateDown), to all the components of the composite
  virtual void setAlignmentPositionError( const AlignmentPositionError& ape, bool propagateDown );

  /// Add the AlignmentPositionError (if this Alignable is a Det) and,
  /// if (propagateDown), add to all the components of the composite
  virtual void addAlignmentPositionError( const AlignmentPositionError& ape, bool propagateDown );

  /// Add the AlignmentPositionError resulting from global rotation (if this Alignable is a Det) and,
  /// if (propagateDown), add to all the components of the composite
  virtual void addAlignmentPositionErrorFromRotation( const RotationType& rotation, bool propagateDown );

  /// Add the AlignmentPositionError resulting from local rotation (if this Alignable is a Det) and,
  /// if (propagateDown), add to all the components of the composite
  virtual void addAlignmentPositionErrorFromLocalRotation( const RotationType& rotation, bool propagateDown );

  /// Set the surface deformation parameters - if (!propagateDown) do not affect daughters
  virtual void setSurfaceDeformation(const SurfaceDeformation *deformation, bool propagateDown);

  /// Add the surface deformation parameters to the existing ones,
  /// if (!propagateDown) do not affect daughters.
  virtual void addSurfaceDeformation(const SurfaceDeformation *deformation, bool propagateDown);

  /// Return the alignable type identifier
  virtual StructureType alignableObjectId() const { return theStructureType; }

  /// Recursive printout of alignable structure
  virtual void dump() const;

  /// Return alignment data
  virtual Alignments* alignments() const;

  /// Return vector of alignment errors
  virtual AlignmentErrorsExtended* alignmentErrors() const;

  /// Return surface deformations
  virtual int surfaceDeformationIdPairs(std::vector<std::pair<int,SurfaceDeformation*> > &) const;

protected:
  /// Constructor from GeomDet, only for use in AlignableDet
  explicit AlignableComposite( const GeomDet* geomDet );

  void setSurface( const AlignableSurface& s) { theSurface = s; }
  
  StructureType theStructureType;

private:
  /// default constructor hidden
  AlignableComposite() : Alignable (0, RotationType()) {};

  Alignables theComponents; // direct daughters
};

#endif
