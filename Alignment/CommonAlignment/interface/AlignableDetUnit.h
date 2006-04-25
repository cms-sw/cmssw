#ifndef Alignment_CommonAlignment_AlignableDetUnit_H
#define Alignment_CommonAlignment_AlignableDetUnit_H

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/CommonDetAlgo/interface/DeepCopyPointer.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

/// A concrete class that allows to (mis)align a GeomDetUnit.
/// Typically all AlignableComposites have (directly or
/// indirectly) this one as the ultimate component.
//
/// Allows for de-/reactivation of the misalignment.

class AlignableDetUnit : public Alignable 
{

public:
  
  AlignableDetUnit( GeomDetUnit* geomDetUnit ) : 
    theGeomDetUnit( geomDetUnit ),
    theOriginalPosition( geomDetUnit->surface().position() ), 
    theOriginalRotation( geomDetUnit->surface().rotation() )  {}

  /// Returns a null vector (no components here)
  virtual std::vector<Alignable*> components() const { return std::vector<Alignable*>(); }

  virtual void move( const GlobalVector& displacement);
	
  /// rotation intepreted such, that the orientation of the rotation
  /// axis is w.r.t. to the global coordinate system, however, the
  /// NOT the center of the rotation is simply taken as the center of
  /// the Alignable-object 
  virtual void rotateInGlobalFrame( const RotationType& rotation);

  virtual GlobalVector displacement() const { return theDisplacement;}

  virtual GlobalPoint globalPosition() const 
  { 
    return theGeomDetUnit->surface().position();
  }

  virtual RotationType globalRotation () const 
  {
    return theGeomDetUnit->surface().rotation();
  }

  virtual const Surface& surface() const { return theGeomDetUnit->surface(); }

  virtual RotationType rotation() const { return theRotation; }
 
  virtual GeomDetUnit* geomDetUnit() const { return theGeomDetUnit; }

  virtual void 
  setAlignmentPositionError(const AlignmentPositionError& ape)
  {
    // Interface only exists at GeomDet level 
    // => static cast (we know GeomDetUnit inherits from GeomDet)
    GeomDet* tmpGeomDet = static_cast<GeomDet*>( theGeomDetUnit );
    DetPositioner::setAlignmentPositionError( *tmpGeomDet, ape );
  }

  virtual void addAlignmentPositionError(const AlignmentPositionError& ape);

  /// Well, any uncertainty in the GLOBAL rotation of a GeomDetUnit currently 
  /// is just taken into acount by an average x,y,z AlignmentPositionError
  virtual void addAlignmentPositionErrorFromRotation(const RotationType& rot);

  /// well, any uncertainty in the LOCAL rotation of a GeomDetUnit currently 
  /// is just taken into acount by an average x,y,z AlignmentPositionError
  virtual void addAlignmentPositionErrorFromLocalRotation(const RotationType& rot);

  /// Restore original position
  virtual void deactivateMisalignment ();

  /// Restore misaligned position
  virtual void reactivateMisalignment ();

  /// Alignable object identifier
  virtual int alignableObjectId () const 
  {
    return AlignableObjectId::AlignableDetUnit; 
  }

private:

  GeomDetUnit* theGeomDetUnit;

  GlobalVector theDisplacement;
  RotationType theRotation;
  //
  // information only available at construction time
  //
  GlobalPoint theOriginalPosition;         ///< position at construction time
  RotationType theOriginalRotation;        ///< rotation at construction time
  //
  // only needed in case deactivation / reactivation is used
  //
  DeepCopyPointer<GlobalPoint> theModifiedPosition;         ///< position saved before deactivation
  DeepCopyPointer<RotationType> theModifiedRotation;        ///< rotation saved before deactivation
  //
  // only needed in case deactivation / reactivation is used
  //
  DeepCopyPointer<GlobalPoint> theReactivatedPosition;      ///< position after reactivation
  DeepCopyPointer<RotationType> theReactivatedRotation;     ///< position after reactivation
};

#endif 



