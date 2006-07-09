#ifndef MeasurementDet_MuonDetLayerMeasurements_H
#define MeasurementDet_MuonDetLayerMeasurements_H

/** \class MuonDetLayerMeasurements
 *  The class to access recHits and TrajectoryMeasurements from DetLayer.  
 *
 *  $Date: 2006/07/06 08:18:27 $
 *  $Revision: 1.8 $
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
class MuonTransientTrackingRecHit;

typedef std::vector<MuonTransientTrackingRecHit*>       RecHitContainer;
typedef std::vector<TrajectoryMeasurement>          MeasurementContainer;
typedef std::pair<const GeomDet*,TrajectoryStateOnSurface> DetWithState;


class MuonDetLayerMeasurements {
 public:
  
  MuonDetLayerMeasurements(bool enableDT = true,
			   bool enableCSC = true,
			   std::string dtlabel = "DTSegment4DProducer", 
			   std::string csclabel = "CSCSegmentProducer");

  virtual ~MuonDetLayerMeasurements();
  
  /// returns TMeasurements in a DetLayer compatible with the TSOS.
  MeasurementContainer
    measurements( const DetLayer* layer,
		  const TrajectoryStateOnSurface& startingState,
		  const Propagator& prop,
		  const MeasurementEstimator& est,
		  const edm::Event& iEvent) const;

  /// faster version in case the TrajectoryState on the surface of the GeomDet is already available
  MeasurementContainer
    fastMeasurements( const DetLayer* layer,
		      const TrajectoryStateOnSurface& theStateOnDet,
		      const TrajectoryStateOnSurface& startingState,
		      const Propagator& prop,
		      const MeasurementEstimator& est,
		      const edm::Event& iEvent) const;

  /// returns TMeasurements in a DetLayer compatible with the TSOS.
  MeasurementContainer
    measurements( const DetLayer* layer,
		  const TrajectoryStateOnSurface& startingState,
		  const Propagator& prop,
		  const MeasurementEstimator& est) const;

  /// faster version in case the TrajectoryState on the surface of the GeomDet is already available
  MeasurementContainer
    fastMeasurements( const DetLayer* layer,
		      const TrajectoryStateOnSurface& theStateOnDet,
		      const TrajectoryStateOnSurface& startingState,
		      const Propagator& prop,
		      const MeasurementEstimator& est) const;

 
  void setEvent(const edm::Event &);  

  /// returns the rechits which are onto the layer
  RecHitContainer recHits(const DetLayer* layer, const edm::Event& iEvent) const;

  /// returns the rechits which are onto the layer
  RecHitContainer recHits(const DetLayer* layer) const;


 private:

  /// obtain TrackingRecHits from a DetLayer
  RecHitContainer recHits(const GeomDet*, const edm::Event& iEvent) const;


  bool enableDTMeasurement;
  bool enableCSCMeasurement;

  std::string theDTRecHitLabel;
  std::string theCSCRecHitLabel;

  bool theEventFlag;
  const edm::Event* theEvent;   
};
#endif

