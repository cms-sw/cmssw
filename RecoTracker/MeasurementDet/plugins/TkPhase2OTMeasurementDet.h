#ifndef TkPhase2OTMeasurementDet_H
#define TkPhase2OTMeasurementDet_H

#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPE.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "RecoTracker/MeasurementDet/src/TkMeasurementDetSet.h"

class TrackingRecHit;
class LocalTrajectoryParameters;

class dso_hidden TkPhase2OTMeasurementDet final : public MeasurementDet {
public:
  typedef edm::Ref<edmNew::DetSetVector<Phase2TrackerCluster1D>, Phase2TrackerCluster1D> Phase2TrackerCluster1DRef;

  typedef edmNew::DetSet<Phase2TrackerCluster1D> detset;
  typedef detset::const_iterator const_iterator;
  typedef ClusterParameterEstimator<Phase2TrackerCluster1D>::LocalValues LocalValues;

  TkPhase2OTMeasurementDet(const GeomDet* gdet, Phase2OTMeasurementConditionSet& conditionSet);

  void update(Phase2OTMeasurementDetSet& data, const detset& detSet) {
    data.update(index(), detSet);
    data.setActiveThisEvent(index(), true);
  }

  void setEmpty(Phase2OTMeasurementDetSet& data) { data.setEmpty(index()); }
  bool isEmpty(const Phase2OTMeasurementDetSet& data) const { return data.empty(index()); }

  ~TkPhase2OTMeasurementDet() override {}

  RecHitContainer recHits(const TrajectoryStateOnSurface&, const MeasurementTrackerEvent& dat) const override;
  bool recHits(const TrajectoryStateOnSurface& stateOnThisDet,
               const MeasurementEstimator&,
               const MeasurementTrackerEvent& data,
               RecHitContainer& result,
               std::vector<float>&) const override;

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

  TrackingRecHit::RecHitPointer buildRecHit(const Phase2TrackerCluster1DRef& cluster,
                                            const LocalTrajectoryParameters& ltp) const;

  /** \brief Turn on/off the module for reconstruction, for the full run or lumi (using info from DB, usually). */
  void setActive(bool active) { conditionSet().setActive(index(), active); }
  /** \brief Turn on/off the module for reconstruction for one events.
             This per-event flag is cleared by any call to 'update' or 'setEmpty'  */
  void setActiveThisEvent(Phase2OTMeasurementDetSet& data, bool active) const {
    data.setActiveThisEvent(index(), active);
  }
  /** \brief Is this module active in reconstruction? It must be both 'setActiveThisEvent' and 'setActive'. */
  bool isActive(const MeasurementTrackerEvent& data) const override { return data.phase2OTData().isActive(index()); }

  bool hasBadComponents(const TrajectoryStateOnSurface& tsos, const MeasurementTrackerEvent& dat) const override;

  //FIXME:just temporary solution for phase2!
  /** \brief Sets the list of bad ROCs, identified by the positions of their centers in the local coordinate frame*/
  //  void setBadRocPositions(std::vector< LocalPoint > & positions) { badRocPositions_.swap(positions); }
  /** \brief Clear the list of bad ROCs */
  //  void clearBadRocPositions() { badRocPositions_.clear(); }

  int index() const { return index_; }
  void setIndex(int i) { index_ = i; }

private:
  unsigned int id_;
  //  std::vector< LocalPoint > badRocPositions_;

  int index_;
  Phase2OTMeasurementConditionSet* theDetConditions;
  Phase2OTMeasurementConditionSet& conditionSet() { return *theDetConditions; }
  const Phase2OTMeasurementConditionSet& conditionSet() const { return *theDetConditions; }

  const ClusterParameterEstimator<Phase2TrackerCluster1D>* cpe() const { return conditionSet().cpe(); }
};

#endif
