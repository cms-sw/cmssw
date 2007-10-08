#ifndef Alignment_CommonAlignment_AlignableDetUnit_H
#define Alignment_CommonAlignment_AlignableDetUnit_H

#include "Alignment/CommonAlignment/interface/Alignable.h"

/// A concrete class that allows to (mis)align a DetUnit.
///
/// Typically all AlignableComposites have (directly or
/// indirectly) this one as the ultimate component.

class AlignableDetUnit : public Alignable 
{

public:
  
  /// Constructor from id and surface
  AlignableDetUnit( align::ID, const AlignableSurface& );
  
  /// Destructor
  virtual ~AlignableDetUnit();

  /// Do nothing (no components here)
  virtual void addComponent( Alignable* ) {}

  /// Returns a null vector (no components here)
  virtual Alignables components() const { return Alignables(); }

  /// Do nothing (no components here, so no subcomponents either...)
  virtual void recursiveComponents(Alignables &result) const {}

  /// Move with respect to the global reference frame
  virtual void move( const GlobalVector& displacement );

  /// Rotation with respect to the global reference frame
  virtual void rotateInGlobalFrame( const RotationType& rotation );

  /// Set the AlignmentPositionError
  virtual void setAlignmentPositionError(const AlignmentPositionError& ape);

  /// Add (or set if it does not exist yet) the AlignmentPositionError
  virtual void addAlignmentPositionError(const AlignmentPositionError& ape);

  /// Add (or set if it does not exist yet) the AlignmentPositionError
  /// resulting from a rotation in the global reference frame
  virtual void addAlignmentPositionErrorFromRotation(const RotationType& rot);

  /// Add (or set if it does not exist yet) the AlignmentPositionError
  /// resulting from a rotation in the local reference frame
  virtual void addAlignmentPositionErrorFromLocalRotation(const RotationType& rot);

  /// Return the alignable type identifier
  virtual StructureType alignableObjectId () const { return align::AlignableDetUnit; }

  /// Printout information about GeomDet
  virtual void dump() const;

  /// Return vector of alignment data
  virtual Alignments* alignments() const;

  /// Return vector of alignment errors
  virtual AlignmentErrors* alignmentErrors() const;

private:

  AlignmentPositionError* theAlignmentPositionError;

};

#endif 
