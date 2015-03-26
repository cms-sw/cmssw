#ifndef Alignment_CommonAlignment_AlignableBeamSpot_h
#define Alignment_CommonAlignment_AlignableBeamSpot_h

/** \class AlignableBeamSpot
 *
 * An Alignable for the beam spot
 *
 *  Original author: Andreas Mussgiller, August 2010
 *
 *  $Date: 2010/10/26 19:53:53 $
 *  $Revision: 1.2 $
 *  (last update by $Author: flucke $)
 */

#include "Alignment/CommonAlignment/interface/Alignable.h"

class SurfaceDeformation;

class AlignableBeamSpot : public Alignable 
{

public:
  
  AlignableBeamSpot();
  
  /// Destructor
  virtual ~AlignableBeamSpot();

  /// Add a component and set its mother to this alignable.
  /// (Note: The component will be adopted, e.g. later deleted.)
  /// Also find average position of this composite from its modules' positions.
  virtual void addComponent( Alignable* component ) {}

  /// Return vector of direct components
  virtual Alignables components() const { return std::vector<Alignable*>(); }

  /// Provide all components, subcomponents etc. (cf. description in base class)
  virtual void recursiveComponents(Alignables &result) const { }

  /// Move with respect to the global reference frame
  virtual void move( const GlobalVector& displacement );

  /// Rotation interpreted in global reference frame
  virtual void rotateInGlobalFrame( const RotationType& rotation );

  /// Set the AlignmentPositionError and, if (propagateDown), to all components
  virtual void setAlignmentPositionError(const AlignmentPositionError &ape, bool propagateDown);

  /// Add (or set if it does not exist yet) the AlignmentPositionError,
  /// if (propagateDown), add also to all components
  virtual void addAlignmentPositionError(const AlignmentPositionError &ape, bool propagateDown);

  /// Add (or set if it does not exist yet) the AlignmentPositionError
  /// resulting from a rotation in the global reference frame,
  /// if (propagateDown), add also to all components
  virtual void addAlignmentPositionErrorFromRotation(const RotationType &rot, bool propagateDown);

  /// Add the AlignmentPositionError resulting from local rotation (if this Alignable is a Det) and,
  /// if (propagateDown), add to all the components of the composite
  virtual void addAlignmentPositionErrorFromLocalRotation( const RotationType& rotation,
							   bool propagateDown );

  /// Return the alignable type identifier
  virtual StructureType alignableObjectId() const { return align::BeamSpot; }

  /// Recursive printout of alignable structure
  virtual void dump() const;

  /// Return vector of alignment data
  virtual Alignments* alignments() const;

  /// Return vector of alignment errors
  virtual AlignmentErrorsExtended* alignmentErrors() const;

  /// alignment position error - for checking only, otherwise use alignmentErrors() above!  
  const AlignmentPositionError* alignmentPositionError() const { return theAlignmentPositionError;}

  /// Return surface deformations
  virtual int surfaceDeformationIdPairs(std::vector<std::pair<int,SurfaceDeformation*> > &) const { return 0; }

  /// do no use, for compatibility only
  virtual void setSurfaceDeformation(const SurfaceDeformation*, bool);
  /// do no use, for compatibility only
  virtual void addSurfaceDeformation(const SurfaceDeformation*, bool);

  /// initialize the alignable with the passed beam spot parameters 
  void initialize(double x, double y, double z,
		  double dxdz, double dydz);

  /// returns the DetId corresponding to the alignable beam spot. Also used
  /// by BeamSpotGeomDet and BeamSpotTransientTrackingRecHit
  static const DetId detId() { return DetId((DetId::Tracker<<DetId::kDetOffset)+0x1ffffff); }

private:

  AlignmentPositionError* theAlignmentPositionError;

  bool theInitializedFlag;
};

#endif // ALIGNABLE_BEAMSPOT_H
