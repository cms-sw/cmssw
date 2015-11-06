#ifndef TkStackMeasurementDet_H
#define TkStackMeasurementDet_H

#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "TkPixelMeasurementDet.h"

#include "Geometry/TrackerGeometryBuilder/interface/StackGeomDet.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"

#include "FWCore/Utilities/interface/Visibility.h"

// FIXME::TkStackMeasurementDet in this moment is just a prototype: to be fixed soon!

class TkStackMeasurementDet GCC11_FINAL : public MeasurementDet {

 public:

  TkStackMeasurementDet( const StackGeomDet* gdet, const PixelClusterParameterEstimator* cpe);
  void init(const MeasurementDet* lowerDet,
	    const MeasurementDet* upperDet);

  virtual RecHitContainer recHits( const TrajectoryStateOnSurface&) const;

  const StackGeomDet& specificGeomDet() const {return static_cast<StackGeomDet const&>(fastGeomDet());}

  virtual bool measurements( const TrajectoryStateOnSurface& stateOnThisDet,
			     const MeasurementEstimator& est,
			     TempMeasurements & result) const;

  const TkPixelMeasurementDet* lowerDet() const{ return theInnerDet;}
  const TkPixelMeasurementDet* upperDet() const{ return theOuterDet;}

  /// return TRUE if both lower and upper components are active
  bool isActive() const {return lowerDet()->isActive() && upperDet()->isActive(); }

  /// return TRUE if at least one of the lower and upper components has badChannels
  bool hasBadComponents( const TrajectoryStateOnSurface &tsos ) const {
    return (lowerDet()->hasBadComponents(tsos) || upperDet()->hasBadComponents(tsos));}

 private:
  const PixelClusterParameterEstimator* thePixelCPE;
  const TkPixelMeasurementDet*       theInnerDet;
  const TkPixelMeasurementDet*       theOuterDet;

};

#endif
