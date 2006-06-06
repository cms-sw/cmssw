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
  AlignableDetUnit( GeomDetUnit* geomDetUnit ) : 
    theGeomDetUnit( geomDetUnit ),
    theOriginalPosition( geomDetUnit->surface().position() ), 
    theOriginalRotation( geomDetUnit->surface().rotation() )  {}

  /// Returns a null vector (no components here)
  virtual std::vector<Alignable*> components() const { return std::vector<Alignable*>(); }

  /// Move with respect to the global reference frame
  virtual void move( const GlobalVector& displacement);

  /// Rotation with respect to the global reference frame
  virtual void rotateInGlobalFrame( const RotationType& rotation);

  /// Return change of global position since the creation of the object 
  virtual const GlobalVector displacement() const { return theDisplacement;}

  /// Return position 
  virtual const GlobalPoint globalPosition() const { return theGeomDetUnit->surface().position(); }

  /// Return orientation with respect to the global reference frame
  virtual const RotationType globalRotation () const { return theGeomDetUnit->surface().rotation(); }

  /// Return the Surface (global position and orientation) of the object
  virtual const Surface& surface() const { return theGeomDetUnit->surface(); }

  /// Return change of orientation since the creation of the object 
  virtual const RotationType rotation() const { return theRotation; }
 
  /// Return corresponding GeomDetUnit
  virtual GeomDetUnit* geomDetUnit() const { return theGeomDetUnit; }

  /// Set the AlignmentPositionError
  virtual void setAlignmentPositionError(const AlignmentPositionError& ape);

  /// Add (or set if it does not exist yet) the AlignmentPositionError
  virtual void addAlignmentPositionError(const AlignmentPositionError& ape);

  /// Add (or set if it does not exist yet) the AlignmentPositionError
  /// resulting from a rotation in the global reference frame
  virtual void addAlignmentPositionErrorFromRotation(const RotationType& rot);

  /// Add (or set if it does not exist yet) the AlignmentPositionError
  /// resulting from a rotation in the global reference frame
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

  GeomDetUnit* theGeomDetUnit;             ///< Associated GeomDetUnit

  GlobalVector theDisplacement;
  RotationType theRotation;

  GlobalPoint theOriginalPosition;         ///< position at construction time
  RotationType theOriginalRotation;        ///< rotation at construction time

  DeepCopyPointer<GlobalPoint>  theModifiedPosition;        ///< position saved before deactivation
  DeepCopyPointer<RotationType> theModifiedRotation;        ///< rotation saved before deactivation

  DeepCopyPointer<GlobalPoint>  theReactivatedPosition;     ///< position after reactivation
  DeepCopyPointer<RotationType> theReactivatedRotation;     ///< position after reactivation

};

#endif 



