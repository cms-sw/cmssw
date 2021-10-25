#ifndef Alignment_TrackerAlignment_TrackerAlignment_H
#define Alignment_TrackerAlignment_TrackerAlignment_H

/** \class TrackerAlignment
 *  The TrackerAlignment helper class for alignment jobs
 *  - Rotates and Translates any module for the tracker based on rawId
 *
 *  \author Nhan Tran
 */

#include "FWCore/Framework/interface/EventSetup.h"

#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
class TrackerTopology;

class TrackerAlignment {
public:
  TrackerAlignment(const TrackerTopology* tTopo, const TrackerGeometry* tGeom);

  ~TrackerAlignment();

  AlignableTracker* getAlignableTracker() { return theAlignableTracker; }

  void moveAlignablePixelEndCaps(int rawId,
                                 const align::Scalars& localDisplacements,
                                 const align::Scalars& localRotations);
  void moveAlignableEndCaps(int rawId, const align::Scalars& localDisplacements, const align::Scalars& localRotations);
  void moveAlignablePixelHalfBarrels(int rawId,
                                     const align::Scalars& localDisplacements,
                                     const align::Scalars& localRotations);
  void moveAlignableInnerHalfBarrels(int rawId,
                                     const align::Scalars& localDisplacements,
                                     const align::Scalars& localRotations);
  void moveAlignableOuterHalfBarrels(int rawId,
                                     const align::Scalars& localDisplacements,
                                     const align::Scalars& localRotations);
  void moveAlignableTIDs(int rawId, const align::Scalars& localDisplacements, const align::Scalars& localRotations);
  void moveAlignableTIBTIDs(int rawId,
                            const align::Scalars& globalDisplacements,
                            const align::RotationType& backwardRotation,
                            const align::RotationType& forwardRotation,
                            bool toAndFro);

  void saveToDB();

private:
  AlignableTracker* theAlignableTracker;
  std::string theAlignRecordName, theErrorRecordName;
};
#endif  //TrackerAlignment_H
