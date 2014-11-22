#ifndef Alignment_CommonAlignment_AlignableDetUnit_H
#define Alignment_CommonAlignment_AlignableDetUnit_H

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"


/// A concrete class that allows to (mis)align a DetUnit.
///
/// Typically all AlignableComposites have (directly or
/// indirectly) this one as the ultimate component.

class AlignableDetUnit : public Alignable 
{

public:
  
  /// Constructor from GeomDetUnit - must not be NULL pointer!
  AlignableDetUnit(const GeomDetUnit *geomDetUnit);
  
  /// Destructor
  virtual ~AlignableDetUnit();

  /// No components here => exception!
  virtual void addComponent( Alignable* );

  /// Returns a null vector (no components here)
  virtual Alignables components() const { return Alignables(); }

  /// Do nothing (no components here, so no subcomponents either...)
  virtual void recursiveComponents(Alignables &result) const {}

  /// Move with respect to the global reference frame
  virtual void move( const GlobalVector& displacement );

  /// Rotation with respect to the global reference frame
  virtual void rotateInGlobalFrame( const RotationType& rotation );

  /// Set the AlignmentPositionError (no components => second argument ignored)
  virtual void setAlignmentPositionError(const AlignmentPositionError &ape, bool /*propDown*/);

  /// Add (or set if it does not exist yet) the AlignmentPositionError
  /// (no components => second argument without effect)
  virtual void addAlignmentPositionError(const AlignmentPositionError& ape, bool /*propDown*/);

  /// Add (or set if it does not exist yet) the AlignmentPositionError
  /// resulting from a rotation in the global reference frame
  /// (no components => second argument without effect)
  virtual void addAlignmentPositionErrorFromRotation(const RotationType& rot, bool /*propDown*/);

  /// Add (or set if it does not exist yet) the AlignmentPositionError
  /// resulting from a rotation in the local reference frame
  /// (no components => second argument without effect)
  virtual void addAlignmentPositionErrorFromLocalRotation(const RotationType& rot, bool /*propDown*/);

  /// Set surface deformation parameters (2nd argument without effect)
  virtual void setSurfaceDeformation(const SurfaceDeformation *deformation, bool);
  /// Add surface deformation parameters to the existing ones (2nd argument without effect)
  virtual void addSurfaceDeformation(const SurfaceDeformation *deformation, bool);

  /// Return the alignable type identifier
  virtual StructureType alignableObjectId () const { return align::AlignableDetUnit; }

  /// Printout information about GeomDet
  virtual void dump() const;

  /// Return vector of alignment data
  virtual Alignments* alignments() const;

  /// Return vector of alignment errors
  virtual AlignmentErrorsExtended* alignmentErrors() const;

  /// Return surface deformations
  virtual int surfaceDeformationIdPairs(std::vector<std::pair<int,SurfaceDeformation*> > &) const;
 
  /// cache the current position, rotation and other parameters (e.g. surface deformations)
  virtual void cacheTransformation();

  /// restore the previously cached transformation
  virtual void restoreCachedTransformation();

  /// alignment position error - for checking only, otherwise use alignmentErrors() above!  
  const AlignmentPositionError* alignmentPositionError() const { return theAlignmentPositionError;}

private:

  AlignmentPositionError* theAlignmentPositionError;
  SurfaceDeformation* theSurfaceDeformation;
  SurfaceDeformation* theCachedSurfaceDeformation;
};

#endif 
