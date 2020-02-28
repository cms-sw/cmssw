#ifndef TkStackMeasurementDet_H
#define TkStackMeasurementDet_H

#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "TkPhase2OTMeasurementDet.h"

#include "Geometry/CommonDetUnit/interface/StackGeomDet.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"

#include "FWCore/Utilities/interface/Visibility.h"

// FIXME::TkStackMeasurementDet in this moment is just a prototype: to be fixed soon!

class TkStackMeasurementDet GCC11_FINAL : public MeasurementDet {
public:
  TkStackMeasurementDet(const StackGeomDet* gdet, const PixelClusterParameterEstimator* cpe);
  void init(const MeasurementDet* lowerDet, const MeasurementDet* upperDet);

  RecHitContainer recHits(const TrajectoryStateOnSurface&, const MeasurementTrackerEvent& data) const override;

  const StackGeomDet& specificGeomDet() const { return static_cast<StackGeomDet const&>(fastGeomDet()); }

  bool measurements(const TrajectoryStateOnSurface& stateOnThisDet,
                    const MeasurementEstimator& est,
                    const MeasurementTrackerEvent& data,
                    TempMeasurements& result) const override;

  const TkPhase2OTMeasurementDet* lowerDet() const { return theInnerDet; }
  const TkPhase2OTMeasurementDet* upperDet() const { return theOuterDet; }

  /// return TRUE if both lower and upper components are active
  bool isActive(const MeasurementTrackerEvent& data) const override {
    return lowerDet()->isActive(data) && upperDet()->isActive(data);
  }

  /// return TRUE if at least one of the lower and upper components has badChannels
  bool hasBadComponents(const TrajectoryStateOnSurface& tsos, const MeasurementTrackerEvent& data) const override {
    return (lowerDet()->hasBadComponents(tsos, data) || upperDet()->hasBadComponents(tsos, data));
  }

private:
  const PixelClusterParameterEstimator* thePixelCPE;
  const TkPhase2OTMeasurementDet* theInnerDet;
  const TkPhase2OTMeasurementDet* theOuterDet;
};

#endif
