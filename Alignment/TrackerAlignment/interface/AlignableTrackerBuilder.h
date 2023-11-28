#ifndef ALIGNMENT_TRACKERALIGNMENT_INTERFACE_ALIGNABLETRACKERBUILDER_H_
#define ALIGNMENT_TRACKERALIGNMENT_INTERFACE_ALIGNABLETRACKERBUILDER_H_

// Original Author:  Max Stark
//         Created:  Thu, 13 Jan 2016 10:22:57 CET

// topology and geometry
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

// alignment
#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/TrackerAlignment/interface/TrackerAlignmentLevelBuilder.h"

class AlignableTrackerBuilder {
  using Alignables = align::Alignables;

  //========================== PUBLIC METHODS =================================
public:  //===================================================================
  AlignableTrackerBuilder(const TrackerGeometry*, const TrackerTopology*);
  virtual ~AlignableTrackerBuilder() = default;

  /// Builds all Alignables (units and composites) of the tracker, based on
  /// the given TrackerGeometry.
  void buildAlignables(AlignableTracker*, bool update = false);

  /// Return tracker name space derived from the tracker's topology
  const align::TrackerNameSpace& trackerNameSpace() const { return trackerAlignmentLevelBuilder_.trackerNameSpace(); }

  /// Return tracker alignable object ID provider  derived from the tracker's geometry
  const AlignableObjectId& objectIdProvider() const { return alignableObjectId_; }

  //========================= PRIVATE METHODS =================================
private:  //==================================================================
  /// Builds Alignables on module-level for each part of the tracker.
  void buildAlignableDetUnits(bool update = false);
  /// Decides whether a GeomDet is from Pixel- or Strip-Detector and calls
  /// the according method to build the Alignable.
  void convertGeomDetsToAlignables(const TrackingGeometry::DetContainer&,
                                   const std::string& moduleName,
                                   bool update = false);
  /// Converts GeomDetUnits of PXB and PXE to AlignableDetUnits.
  void buildPixelDetectorAlignable(
      const GeomDet*, int subdetId, Alignables& aliDets, Alignables& aliDetUnits, bool update = false);
  /// Converts GeomDets of TIB, TID, TOB and TEC either to AlignableDetUnits
  /// or AlignableSiStripDet, depending on the module-type (2D or 1D).
  void buildStripDetectorAlignable(
      const GeomDet*, int subdetId, Alignables& aliDets, Alignables& aliDetUnits, bool update = false);

  void buildOuterTrackerDetectorAlignable(
      const GeomDet*, int subdetId, Alignables& aliDets, Alignables& aliDetUnits, bool update = false);

  /// Builds all composite Alignables for the tracker. The hierarchy and
  /// numbers of components are determined in TrackerAlignmentLevelBuilder.
  void buildAlignableComposites(bool update = false);
  /// Builds the PixelDetector by hand.
  void buildPixelDetector(AlignableTracker*);
  /// Builds the StripDetector by hand.
  void buildStripDetector(AlignableTracker*);
  /// Builds the Phase-2 Outer Tracker Detector by hand.
  void buildOuterTrackerDetector(AlignableTracker*);

  //========================== PRIVATE DATA ===================================
  //===========================================================================

  const TrackerGeometry* trackerGeometry_;
  const TrackerTopology* trackerTopology_;
  const AlignableObjectId alignableObjectId_;

  AlignableMap* alignableMap_;

  TrackerAlignmentLevelBuilder trackerAlignmentLevelBuilder_;

  int numDetUnits = 0;
};

#endif /* ALIGNMENT_TRACKERALIGNMENT_INTERFACE_ALIGNABLETRACKERBUILDER_H_ */
