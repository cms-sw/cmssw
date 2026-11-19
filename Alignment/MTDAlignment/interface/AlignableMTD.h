#ifndef Alignment_MTDAlignment_AlignableMTD_H
#define Alignment_MTDAlignment_AlignableMTD_H

/** \class AlignableMTD
 *  The alignable MTD
 *
 *  $Date: 2024/12/15 21:23:15 $
 *  $Revision: 1.00 $
 *  \author Pablo Martinez Ruiz del Arbol - IFCA
 */

#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include <DataFormats/GeometryVector/interface/GlobalPoint.h>

#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"

class MTDGeometry;

// Classes that will be used to construct the MTD
class AlignableBTL;
class AlignableBTLTray;
class AlignableBTLRU;
class AlignableBTLModule;
class AlignableBTLSensorModule;
class AlignableETLEndcap;
class AlignableETLDisk;
class AlignableETLDee;
class AlignableETLServiceHybrid;
class AlignableETLModule;
class AlignableETLSensor;

/// Constructor of the full MTD geometry.

class AlignableMTD : public AlignableComposite {
public:
  /// Constructor from geometries
  AlignableMTD(const MTDGeometry*);

  /// Destructor
  ~AlignableMTD() override;

  /// Updater using MTDGeometry.
  /// The given geometries have to match the current ones.
  void update(const MTDGeometry*);

  /// Return all components
  const align::Alignables& components() const final { return theMTDComponents; }

  /// Alignable tracker has no mother
  virtual Alignable* mother() { return nullptr; }

  /// Methods to return specific of components
  align::Alignables BTLSensorModules();
  align::Alignables BTLModules();
  align::Alignables BTLRUs();
  align::Alignables BTLTrays();
  align::Alignables BTLBarrel();
  align::Alignables ETLEndcaps();
  align::Alignables ETLDisks();
  align::Alignables ETLServiceHybrids();
  align::Alignables ETLDees();
  align::Alignables ETLModules();
  align::Alignables ETLSensors();

  /// Get MTD alignments sorted by DetId
  Alignments* mtdAlignments();

  /// Get MTD alignments errors sorted by DetId
  AlignmentErrorsExtended* mtdAlignmentErrorsExtended();

  /// Get BTL alignments sorted by DetId
  Alignments* btlAlignments();

  /// Get BTL alignment errors sorted by DetId
  AlignmentErrorsExtended* btlAlignmentErrorsExtended();

  /// Get ETL alignments sorted by DetId
  Alignments* etlAlignments();

  /// Get ETL alignment errors sorted by DetId
  AlignmentErrorsExtended* etlAlignmentErrorsExtended();

  /// Return MTD alignable object ID provider derived from the MTD system geometry
  const AlignableObjectId& objectIdProvider() const { return alignableObjectId_; }

private:
  /// Get the position (centered at 0 by default)
  PositionType computePosition();

  /// Get the global orientation (no rotation by default)
  RotationType computeOrientation();

  /// Get the Surface
  AlignableSurface computeSurface();

  /// Get alignments sorted by DetId
  Alignments* alignments() const override;

  /// Get alignment errors sorted by DetId
  AlignmentErrorsExtended* alignmentErrors() const override;

  // Sub-structure builders

  /// Build BTL
  void buildBTLBarrel(const MTDGeometry*, bool update = false);

  /// Build ETL
  void buildETLEndcap(const MTDGeometry*, bool update = false);

  /// Set mothers recursively
  void recursiveSetMothers(Alignable* alignable);

  /// alignable object ID provider
  const AlignableObjectId alignableObjectId_;

  /// Containers of separate components
  std::vector<AlignableBTLSensorModule*> theBTLSensorModules;
  std::vector<AlignableBTLModule*> theBTLModules;
  std::vector<AlignableBTLRU*> theBTLRUs;
  std::vector<AlignableBTLTray*> theBTLTrays;
  std::vector<AlignableBTL*> theBTLBarrel;
  std::vector<AlignableETLEndcap*> theETLEndcap;
  std::vector<AlignableETLDisk*> theETLDisks;
  std::vector<AlignableETLDee*> theETLDees;
  std::vector<AlignableETLServiceHybrid*> theETLServiceHybrids;
  std::vector<AlignableETLModule*> theETLModules;
  std::vector<AlignableETLSensor*> theETLSensors;
  align::Alignables theMTDComponents;
};

#endif  //AlignableMTD_H
