#ifndef TkPixelMeasurementDet_H
#define TkPixelMeasurementDet_H

#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
//#include "DataFormats/SiPixelCluster/interface/SiPixelClusterFwd.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "RecoTracker/MeasurementDet/src/TkMeasurementDetSet.h"

class TrackingRecHit;
class LocalTrajectoryParameters;

class dso_hidden TkPixelMeasurementDet final : public MeasurementDet {
public:
  typedef edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster> SiPixelClusterRef;

  typedef edmNew::DetSet<SiPixelCluster> detset;
  typedef detset::const_iterator const_iterator;
  typedef PixelClusterParameterEstimator::LocalValues LocalValues;

  TkPixelMeasurementDet(const GeomDet* gdet, PxMeasurementConditionSet& conditionSet);

  void update(PxMeasurementDetSet& data, const detset& detSet) {
    data.update(index(), detSet);
    data.setActiveThisEvent(index(), true);
  }

  void setEmpty(PxMeasurementDetSet& data) { data.setEmpty(index()); }
  bool isEmpty(const PxMeasurementDetSet& data) const { return data.empty(index()); }

  ~TkPixelMeasurementDet() override {}

  // all hits
  RecHitContainer recHits(const TrajectoryStateOnSurface&, const MeasurementTrackerEvent& dat) const override;

  // only hits compatible with tsos
  RecHitContainer compHits(const TrajectoryStateOnSurface&,
                           const MeasurementTrackerEvent& dat,
                           float xl,
                           float yl) const;

  // simple hits
  bool recHits(SimpleHitContainer& result,
               const TrajectoryStateOnSurface& stateOnThisDet,
               const MeasurementEstimator&,
               const MeasurementTrackerEvent& data) const override {
    assert("not implemented for Pixel yet" == nullptr);
  }

  bool measurements(const TrajectoryStateOnSurface& stateOnThisDet,
                    const MeasurementEstimator& est,
                    const MeasurementTrackerEvent& dat,
                    TempMeasurements& result) const override;

  const PixelGeomDetUnit& specificGeomDet() const { return static_cast<PixelGeomDetUnit const&>(fastGeomDet()); }

  TrackingRecHit::RecHitPointer buildRecHit(const SiPixelClusterRef& cluster,
                                            const LocalTrajectoryParameters& ltp) const;

  /** \brief Turn on/off the module for reconstruction, for the full run or lumi (using info from DB, usually). */
  void setActive(bool active) { conditionSet().setActive(index(), active); }
  /** \brief Turn on/off the module for reconstruction for one events.
             This per-event flag is cleared by any call to 'update' or 'setEmpty'  */
  void setActiveThisEvent(PxMeasurementDetSet& data, bool active) const { data.setActiveThisEvent(index(), active); }
  /** \brief Is this module active in reconstruction? It must be both 'setActiveThisEvent' and 'setActive'. */
  bool isActive(const MeasurementTrackerEvent& data) const override { return data.pixelData().isActive(index()); }

  bool hasBadComponents(const TrajectoryStateOnSurface& tsos, const MeasurementTrackerEvent& dat) const override;

  /** \brief Sets the list of bad ROCs, identified by the positions of their centers in the local coordinate frame*/
  void setBadRocPositions(std::vector<LocalPoint>& positions) { badRocPositions_.swap(positions); }
  /** \brief Clear the list of bad ROCs */
  void clearBadRocPositions() { badRocPositions_.clear(); }

  const PxMeasurementDetSet::BadFEDChannelPositions* getBadFEDChannelPositions(
      const MeasurementTrackerEvent& data) const {
    return data.pixelData().getBadFEDChannelPositions(index());
  }

  int index() const { return index_; }
  void setIndex(int i) { index_ = i; }

private:
  unsigned int id_{};
  std::vector<LocalPoint> badRocPositions_;

  int index_{};
  PxMeasurementConditionSet* theDetConditions;
  PxMeasurementConditionSet& conditionSet() { return *theDetConditions; }
  const PxMeasurementConditionSet& conditionSet() const { return *theDetConditions; }

  const PixelClusterParameterEstimator* cpe() const { return conditionSet().pixelCPE(); }

public:
  inline bool accept(SiPixelClusterRefNew& r, const std::vector<bool> skipClusters) const {
    if (skipClusters.empty())
      return true;
    if (r.key() >= skipClusters.size()) {
      edm::LogError("IndexMisMatch") << r.key() << " is larger than: " << skipClusters.size() << " no skipping done";
      return true;
    }
    return not skipClusters[r.key()];
  }
};

#endif
