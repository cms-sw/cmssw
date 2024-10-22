#ifndef Alignment_CommonAlignment_AlignableDetUnit_H
#define Alignment_CommonAlignment_AlignableDetUnit_H

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

/// A concrete class that allows to (mis)align a DetUnit.
///
/// Typically all AlignableComposites have (directly or
/// indirectly) this one as the ultimate component.

class AlignableDetUnit : public Alignable {
public:
  /// Constructor from GeomDetUnit - must not be NULL pointer!
  AlignableDetUnit(const GeomDetUnit* geomDetUnit);

  /// Destructor
  ~AlignableDetUnit() override;

  /// Updater from GeomDetUnit
  /// The given GeomDetUnit id has to match the current id.
  void update(const GeomDetUnit* geomDetUnit);

  /// No components here => exception!
  void addComponent(Alignable*) final;

  /// Returns a null vector (no components here)
  const Alignables& components() const override { return emptyComponents_; }

  /// Do nothing (no components here, so no subcomponents either...)
  void recursiveComponents(Alignables& result) const override {}

  /// Move with respect to the global reference frame
  void move(const GlobalVector& displacement) override;

  /// Rotation with respect to the global reference frame
  void rotateInGlobalFrame(const RotationType& rotation) override;

  /// Set the AlignmentPositionError (no components => second argument ignored)
  void setAlignmentPositionError(const AlignmentPositionError& ape, bool /*propDown*/) final;

  /// Add (or set if it does not exist yet) the AlignmentPositionError
  /// (no components => second argument without effect)
  void addAlignmentPositionError(const AlignmentPositionError& ape, bool /*propDown*/) final;

  /// Add (or set if it does not exist yet) the AlignmentPositionError
  /// resulting from a rotation in the global reference frame
  /// (no components => second argument without effect)
  void addAlignmentPositionErrorFromRotation(const RotationType& rot, bool /*propDown*/) final;

  /// Add (or set if it does not exist yet) the AlignmentPositionError
  /// resulting from a rotation in the local reference frame
  /// (no components => second argument without effect)
  void addAlignmentPositionErrorFromLocalRotation(const RotationType& rot, bool /*propDown*/) final;

  /// Set surface deformation parameters (2nd argument without effect)
  void setSurfaceDeformation(const SurfaceDeformation* deformation, bool) final;
  /// Add surface deformation parameters to the existing ones (2nd argument without effect)
  void addSurfaceDeformation(const SurfaceDeformation* deformation, bool) final;

  /// Return the alignable type identifier
  StructureType alignableObjectId() const override { return align::AlignableDetUnit; }

  /// Printout information about GeomDet
  void dump() const override;

  /// Return vector of alignment data
  Alignments* alignments() const override;

  /// Return vector of alignment errors
  AlignmentErrorsExtended* alignmentErrors() const override;

  /// Return surface deformations
  int surfaceDeformationIdPairs(std::vector<std::pair<int, SurfaceDeformation*> >&) const override;

  /// cache the current position, rotation and other parameters (e.g. surface deformations)
  void cacheTransformation() override;

  /// cache for the given run the current position, rotation and other parameters (e.g. surface deformations)
  void cacheTransformation(const align::RunNumber&) override;

  /// restore the previously cached transformation
  void restoreCachedTransformation() override;

  /// restore for the given run the previously cached transformation
  void restoreCachedTransformation(const align::RunNumber&) override;

  /// alignment position error - for checking only, otherwise use alignmentErrors() above!
  const AlignmentPositionError* alignmentPositionError() const { return theAlignmentPositionError; }

private:
  static const Alignables emptyComponents_;
  AlignmentPositionError* theAlignmentPositionError;
  SurfaceDeformation* theSurfaceDeformation;
  SurfaceDeformation* theCachedSurfaceDeformation;
  Cache<SurfaceDeformation*> surfaceDeformationsCache_;
};

#endif
