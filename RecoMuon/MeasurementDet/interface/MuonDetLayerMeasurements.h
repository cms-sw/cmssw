#ifndef MeasurementDet_MuonDetLayerMeasurements_H
#define MeasurementDet_MuonDetLayerMeasurements_H

/** \class MuonDetLayerMeasurements
 *  The class to access recHits and TrajectoryMeasurements from DetLayer.  
 *
 *  $Date: $
 *  $Revision: $
 *  \author C. Liu - Purdue University
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"

#include <vector>

class DetLayer;
class GeomDet;
class TrajectoryMeasurement;
class TransientTrackingRecHit;

typedef std::vector<TransientTrackingRecHit*>       RecHitContainer;
typedef std::vector<TrajectoryMeasurement>          MeasurementContainer;

class MuonDetLayerMeasurements {
public:

   MuonDetLayerMeasurements();

   virtual ~MuonDetLayerMeasurements();

   /// obtain TrackingRecHits from a DetLayer
   RecHitContainer recHits(const DetLayer* layer, const edm::Event& iEvent) const;

   /// returns TMeasurements in a DetLayer compatible with the TSOS.
   MeasurementContainer
   measurements( const DetLayer& layer,
                 const TrajectoryStateOnSurface& startingState,
                 const Propagator& prop,
                 const MeasurementEstimator& est,
                 const edm::Event& iEvent) const;

 /// faster version in case the TrajectoryState on the surface of the GeomDet is already available
   MeasurementContainer
   fastMeasurements( const DetLayer& layer,
                     const TrajectoryStateOnSurface& theStateOnDet,
                     const TrajectoryStateOnSurface& startingState,
                     const Propagator& prop,
                     const MeasurementEstimator& est,
                     const edm::Event& iEvent) const;

private:
};
#endif

