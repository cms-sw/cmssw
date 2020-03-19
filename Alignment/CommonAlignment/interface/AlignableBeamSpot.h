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

class AlignableBeamSpot : public Alignable {
public:
  AlignableBeamSpot();

  /// Destructor
  ~AlignableBeamSpot() override;

  /// Add a component and set its mother to this alignable.
  /// (Note: The component will be adopted, e.g. later deleted.)
  /// Also find average position of this composite from its modules' positions.
  void addComponent(Alignable* component) override {}

  /// Return vector of direct components
  const Alignables& components() const override { return emptyComponents_; }

  /// Provide all components, subcomponents etc. (cf. description in base class)
  void recursiveComponents(Alignables& result) const override {}

  /// Move with respect to the global reference frame
  void move(const GlobalVector& displacement) override;

  /// Rotation interpreted in global reference frame
  void rotateInGlobalFrame(const RotationType& rotation) override;

  /// Set the AlignmentPositionError and, if (propagateDown), to all components
  void setAlignmentPositionError(const AlignmentPositionError& ape, bool propagateDown) override;

  /// Add (or set if it does not exist yet) the AlignmentPositionError,
  /// if (propagateDown), add also to all components
  void addAlignmentPositionError(const AlignmentPositionError& ape, bool propagateDown) override;

  /// Add (or set if it does not exist yet) the AlignmentPositionError
  /// resulting from a rotation in the global reference frame,
  /// if (propagateDown), add also to all components
  void addAlignmentPositionErrorFromRotation(const RotationType& rot, bool propagateDown) override;

  /// Add the AlignmentPositionError resulting from local rotation (if this Alignable is a Det) and,
  /// if (propagateDown), add to all the components of the composite
  void addAlignmentPositionErrorFromLocalRotation(const RotationType& rotation, bool propagateDown) override;

  /// Return the alignable type identifier
  StructureType alignableObjectId() const override { return align::BeamSpot; }

  /// Recursive printout of alignable structure
  void dump() const override;

  /// Return vector of alignment data
  Alignments* alignments() const override;

  /// Return vector of alignment errors
  AlignmentErrorsExtended* alignmentErrors() const override;

  /// alignment position error - for checking only, otherwise use alignmentErrors() above!
  const AlignmentPositionError* alignmentPositionError() const { return theAlignmentPositionError; }

  /// Return surface deformations
  int surfaceDeformationIdPairs(std::vector<std::pair<int, SurfaceDeformation*> >&) const override { return 0; }

  /// do no use, for compatibility only
  void setSurfaceDeformation(const SurfaceDeformation*, bool) override;
  /// do no use, for compatibility only
  void addSurfaceDeformation(const SurfaceDeformation*, bool) override;

  /// initialize the alignable with the passed beam spot parameters
  void initialize(double x, double y, double z, double dxdz, double dydz);

  /// reset beam spot to the uninitialized state
  void reset();

  /// returns the DetId corresponding to the alignable beam spot. Also used
  /// by BeamSpotGeomDet and BeamSpotTransientTrackingRecHit
  static const DetId detId() { return DetId((DetId::Tracker << DetId::kDetOffset) + 0x1ffffff); }

private:
  static const Alignables emptyComponents_;
  AlignmentPositionError* theAlignmentPositionError;

  bool theInitializedFlag;
};

#endif  // ALIGNABLE_BEAMSPOT_H
