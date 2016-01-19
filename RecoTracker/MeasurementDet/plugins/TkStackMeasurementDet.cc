#include "TkStackMeasurementDet.h"

#include "TrackingTools/MeasurementDet/interface/MeasurementDetException.h"

using namespace std;

TkStackMeasurementDet::TkStackMeasurementDet( const StackGeomDet* gdet,
                                              const PixelClusterParameterEstimator* cpe) :
  MeasurementDet(gdet),
  thePixelCPE(cpe),
  theInnerDet(nullptr), theOuterDet(nullptr)
{}

void TkStackMeasurementDet::init(const MeasurementDet* lowerDet,
                                 const MeasurementDet* upperDet) {
  theInnerDet = dynamic_cast<const TkPixelMeasurementDet *>(lowerDet);
  theOuterDet = dynamic_cast<const TkPixelMeasurementDet *>(upperDet);

  if ((theInnerDet == 0) || (theOuterDet == 0)) {
    throw MeasurementDetException("TkStackMeasurementDet ERROR: Trying to glue a det which is not a TkPixelMeasurementDet");
  }
}

TkStackMeasurementDet::RecHitContainer
TkStackMeasurementDet::recHits( const TrajectoryStateOnSurface& ts, const MeasurementTrackerEvent & data) const
{
  RecHitContainer result;
/*
  HitCollectorForRecHits collector( &fastGeomDet(), theMatcher, theCPE, result );
  collectRecHits(ts, collector);
*/
  return result;
}

bool TkStackMeasurementDet::measurements( const TrajectoryStateOnSurface& stateOnThisDet,
                                          const MeasurementEstimator& est, const MeasurementTrackerEvent & data,
                                          TempMeasurements & result) const {
  return true;
}
