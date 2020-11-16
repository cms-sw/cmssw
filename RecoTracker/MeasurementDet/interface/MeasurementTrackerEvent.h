#ifndef MeasurementTrackerEvent_H
#define MeasurementTrackerEvent_H

#include <vector>
class StMeasurementDetSet;
class PxMeasurementDetSet;
class Phase2OTMeasurementDetSet;
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/TrackerRecHit2D/interface/VectorHit.h"
#include "DataFormats/Common/interface/ContainerMask.h"

#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"

class MeasurementTrackerEvent {
public:
  using QualityFlags = MeasurementTracker::QualityFlags;

  /// Dummy constructor used for I/O (even if it's a transient object)
  MeasurementTrackerEvent() {}
  ~MeasurementTrackerEvent();

  /// Real constructor 1: with the full data (owned)
  MeasurementTrackerEvent(const MeasurementTracker &tracker,
                          const StMeasurementDetSet *strips,
                          const PxMeasurementDetSet *pixels,
                          const Phase2OTMeasurementDetSet *phase2OT,
                          const VectorHitCollection *phase2OTVectorHits,
                          const VectorHitCollection *phase2OTVectorHitsRej,
                          const std::vector<bool> &stripClustersToSkip,
                          const std::vector<bool> &pixelClustersToSkip,
                          const std::vector<bool> &phase2OTClustersToSkip)
      : theTracker(&tracker),
        theStripData(strips),
        thePixelData(pixels),
        thePhase2OTData(phase2OT),
        thePhase2OTVectorHits(phase2OTVectorHits),
        thePhase2OTVectorHitsRej(phase2OTVectorHitsRej),
        theOwner(true),
        theStripClustersToSkip(stripClustersToSkip),
        thePixelClustersToSkip(pixelClustersToSkip),
        thePhase2OTClustersToSkip(phase2OTClustersToSkip) {}

  /// Real constructor 2: with new cluster skips (checked)
  MeasurementTrackerEvent(const MeasurementTrackerEvent &trackerEvent,
                          const edm::ContainerMask<edmNew::DetSetVector<SiStripCluster> > &stripClustersToSkip,
                          const edm::ContainerMask<edmNew::DetSetVector<SiPixelCluster> > &pixelClustersToSkip);

  //FIXME:just temporary solution for phase2!
  MeasurementTrackerEvent(
      const MeasurementTrackerEvent &trackerEvent,
      const edm::ContainerMask<edmNew::DetSetVector<SiPixelCluster> > &phase2pixelClustersToSkip,
      const edm::ContainerMask<edmNew::DetSetVector<Phase2TrackerCluster1D> > &phase2OTClustersToSkip);

  MeasurementTrackerEvent(const MeasurementTrackerEvent &other) = delete;
  MeasurementTrackerEvent &operator=(const MeasurementTrackerEvent &other) = delete;
  MeasurementTrackerEvent(MeasurementTrackerEvent &&other);
  MeasurementTrackerEvent &operator=(MeasurementTrackerEvent &&other);

  const MeasurementTracker &measurementTracker() const { return *theTracker; }
  const StMeasurementDetSet &stripData() const { return *theStripData; }
  const PxMeasurementDetSet &pixelData() const { return *thePixelData; }
  const Phase2OTMeasurementDetSet &phase2OTData() const { return *thePhase2OTData; }
  const VectorHitCollection &phase2OTVectorHits() const { return *thePhase2OTVectorHits; }
  const VectorHitCollection &phase2OTVectorHitsRej() const { return *thePhase2OTVectorHitsRej; }
  const std::vector<bool> &stripClustersToSkip() const { return theStripClustersToSkip; }
  const std::vector<bool> &pixelClustersToSkip() const { return thePixelClustersToSkip; }
  const std::vector<bool> &phase2OTClustersToSkip() const { return thePhase2OTClustersToSkip; }

  // forwarded calls
  const TrackerGeometry *geomTracker() const { return measurementTracker().geomTracker(); }
  const GeometricSearchTracker *geometricSearchTracker() const { return measurementTracker().geometricSearchTracker(); }

  /// Previous MeasurementDetSystem interface
  MeasurementDetWithData idToDet(const DetId &id) const { return measurementTracker().idToDet(id, *this); }

private:
  const MeasurementTracker *theTracker = nullptr;
  const StMeasurementDetSet *theStripData = nullptr;
  const PxMeasurementDetSet *thePixelData = nullptr;
  const Phase2OTMeasurementDetSet *thePhase2OTData = nullptr;
  const VectorHitCollection *thePhase2OTVectorHits = nullptr;
  const VectorHitCollection *thePhase2OTVectorHitsRej = nullptr;
  bool theOwner = false;  // do I own the tree above?
  // these  could be const pointers as well, but ContainerMask doesn't expose the vector
  std::vector<bool> theStripClustersToSkip;
  std::vector<bool> thePixelClustersToSkip;
  std::vector<bool> thePhase2OTClustersToSkip;
};

#endif  // MeasurementTrackerEvent_H
