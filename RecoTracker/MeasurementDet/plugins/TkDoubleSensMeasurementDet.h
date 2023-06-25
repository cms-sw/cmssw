#ifndef TkDoubleSensMeasurementDet_H
#define TkDoubleSensMeasurementDet_H

#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "TkPixelMeasurementDet.h"

#include "Geometry/CommonDetUnit/interface/StackGeomDet.h"
#include "Geometry/CommonDetUnit/interface/DoubleSensGeomDet.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"

#include "FWCore/Utilities/interface/Visibility.h"

class TkDoubleSensMeasurementDet final : public MeasurementDet {
public:
  TkDoubleSensMeasurementDet(const DoubleSensGeomDet* gdet, const PixelClusterParameterEstimator* cpe);

  void init(const MeasurementDet* firstDet, const MeasurementDet* secondDet);

  RecHitContainer recHits(const TrajectoryStateOnSurface&, const MeasurementTrackerEvent& data) const override;

  const DoubleSensGeomDet& specificGeomDet() const { return static_cast<DoubleSensGeomDet const&>(fastGeomDet()); }
  typedef edm::Ref<edmNew::DetSetVector<Phase2TrackerCluster1D>, Phase2TrackerCluster1D> Phase2TrackerCluster1DRef;

  typedef edmNew::DetSet<SiPixelCluster> detset;
  typedef detset::const_iterator const_iterator;

  bool measurements(const TrajectoryStateOnSurface& stateOnThisDet,
                    const MeasurementEstimator& est,
                    const MeasurementTrackerEvent& data,
                    TempMeasurements& result) const override;

  const TkPixelMeasurementDet* firstDet() const { return theFirstDet; }
  const TkPixelMeasurementDet* secondDet() const { return theSecondDet; }

  /// return TRUE if both first and second components are active
  bool isActive(const MeasurementTrackerEvent& data) const override {
    return firstDet()->isActive(data) && secondDet()->isActive(data);
  }
  bool isEmpty(const PxMeasurementDetSet& data) const {
    return data.empty(firstDet()->index()) || data.empty(secondDet()->index());
  }

  /// return TRUE if at least one of the first and second components has badChannels
  bool hasBadComponents(const TrajectoryStateOnSurface& tsos, const MeasurementTrackerEvent& data) const override {
    return (firstDet()->hasBadComponents(tsos, data) || secondDet()->hasBadComponents(tsos, data));
  }

private:
  const PixelClusterParameterEstimator* thePixelCPE;
  const TkPixelMeasurementDet* theFirstDet;
  const TkPixelMeasurementDet* theSecondDet;
};

#endif
