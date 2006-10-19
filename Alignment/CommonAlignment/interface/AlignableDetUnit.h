#ifndef Alignment_CommonAlignment_AlignableDetUnit_H
#define Alignment_CommonAlignment_AlignableDetUnit_H

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"

#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/CommonDetAlgo/interface/DeepCopyPointer.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

/// A concrete class that allows to (mis)align a GeomDetUnit.
///
/// Typically all AlignableComposites have (directly or
/// indirectly) this one as the ultimate component.
/// Allows for de-/reactivation of the misalignment.

class AlignableDetUnit : public Alignable 
{

public:
  
  /// Constructor from GeomDetUnits
  AlignableDetUnit( const GeomDetUnit* geomDetUnit );
  
  /// Destructor
  ~AlignableDetUnit();

  /// Returns a null vector (no components here)
  virtual std::vector<Alignable*> components() const { return std::vector<Alignable*>(); }
  /// Do nothing (no components here, so no subcomponents either...)
  virtual void recursiveComponents(std::vector<Alignable*> &result) const;


  /// Move with respect to the global reference frame
  virtual void move( const GlobalVector& displacement );

  /// Rotation with respect to the global reference frame
  virtual void rotateInGlobalFrame( const RotationType& rotation );

  /// Return position 
  virtual const GlobalPoint globalPosition() const { return theSurface.position(); }

  /// Return orientation with respect to the global reference frame
  virtual const RotationType globalRotation () const { return theSurface.rotation(); }

  /// Return the Surface (global position and orientation) of the object
  virtual const AlignableSurface& surface() const { return theSurface; }

  /// Return corresponding GeomDetUnit ID
  virtual DetId geomDetUnitId() const { return theDetUnitId; }

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

  /// Restore original position
  virtual void deactivateMisalignment ();

  /// Restore misaligned position
  virtual void reactivateMisalignment ();

  /// Return the alignable type identifier
  virtual int alignableObjectId () const { return AlignableObjectId::AlignableDetUnit; }

  /// Printout information about GeomDet
  virtual void dump() const;

  /// Return vector of alignment data
  virtual Alignments* alignments() const;

  /// Return vector of alignment errors
  virtual AlignmentErrors* alignmentErrors() const;

private:

  DetId theDetUnitId;             ///< Associated GeomDetUnit Id

  AlignableSurface theOriginalSurface;
  AlignableSurface theSurface;
  AlignableSurface theSavedSurface;

  float theWidth, theLength;

  AlignmentPositionError* theAlignmentPositionError;

};

#endif 



