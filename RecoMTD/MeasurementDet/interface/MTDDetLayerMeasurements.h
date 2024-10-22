#ifndef RecoMTD_MeasurementDet_MTDDetLayerMeasurements_H
#define RecoMTD_MeasurementDet_MTDDetLayerMeasurements_H

/** \class MTDDetLayerMeasurements
 *  The class to access recHits and TrajectoryMeasurements from DetLayer.  
 *
 *  \author B. Tannenwald 
 *  Adapted from RecoMuon version.
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"
#include "TrackingTools/MeasurementDet/interface/TrajectoryMeasurementGroup.h"
#include "DataFormats/TrackerRecHit2D/interface/MTDTrackingRecHit.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include <vector>

class DetLayer;
class GeomDet;
class TrajectoryMeasurement;

class MTDDetLayerMeasurements {
public:
  typedef std::vector<TrajectoryMeasurement> MeasurementContainer;
  typedef std::pair<const GeomDet*, TrajectoryStateOnSurface> DetWithState;
  typedef std::vector<GenericTransientTrackingRecHit::RecHitPointer> MTDRecHitContainer;

  MTDDetLayerMeasurements(const edm::InputTag& mtdlabel, edm::ConsumesCollector& iC);

  virtual ~MTDDetLayerMeasurements();

  // for a given det and state.  Not clear when the fastMeasurements below
  //  should be used, since it isn't passed a GeomDet
  MeasurementContainer measurements(const DetLayer* layer,
                                    const GeomDet* det,
                                    const TrajectoryStateOnSurface& stateOnDet,
                                    const MeasurementEstimator& est,
                                    const edm::Event& iEvent);

  /// returns TMeasurements in a DetLayer compatible with the TSOS.
  MeasurementContainer measurements(const DetLayer* layer,
                                    const TrajectoryStateOnSurface& startingState,
                                    const Propagator& prop,
                                    const MeasurementEstimator& est,
                                    const edm::Event& iEvent);

  /// faster version in case the TrajectoryState on the surface of the GeomDet is already available
  MeasurementContainer fastMeasurements(const DetLayer* layer,
                                        const TrajectoryStateOnSurface& theStateOnDet,
                                        const TrajectoryStateOnSurface& startingState,
                                        const Propagator& prop,
                                        const MeasurementEstimator& est,
                                        const edm::Event& iEvent);

  /// returns TMeasurements in a DetLayer compatible with the TSOS.
  MeasurementContainer measurements(const DetLayer* layer,
                                    const TrajectoryStateOnSurface& startingState,
                                    const Propagator& prop,
                                    const MeasurementEstimator& est);

  /// faster version in case the TrajectoryState on the surface of the GeomDet is already available
  MeasurementContainer fastMeasurements(const DetLayer* layer,
                                        const TrajectoryStateOnSurface& theStateOnDet,
                                        const TrajectoryStateOnSurface& startingState,
                                        const Propagator& prop,
                                        const MeasurementEstimator& est);

  std::vector<TrajectoryMeasurementGroup> groupedMeasurements(const DetLayer* layer,
                                                              const TrajectoryStateOnSurface& startingState,
                                                              const Propagator& prop,
                                                              const MeasurementEstimator& est,
                                                              const edm::Event& iEvent);

  std::vector<TrajectoryMeasurementGroup> groupedMeasurements(const DetLayer* layer,
                                                              const TrajectoryStateOnSurface& startingState,
                                                              const Propagator& prop,
                                                              const MeasurementEstimator& est);

  void setEvent(const edm::Event&);

  /// returns the rechits which are on the layer
  MTDRecHitContainer recHits(const DetLayer* layer, const edm::Event& iEvent);

  /// returns the rechits which are on the layer
  MTDRecHitContainer recHits(const DetLayer* layer);

private:
  /// obtain TrackingRecHits from a DetLayer
  MTDRecHitContainer recHits(const GeomDet*, const edm::Event& iEvent);

  /// check that the event is set, and throw otherwise
  void checkEvent() const;

  // sort functions for containers provided by measurements methods
  template <class T>
  T sortResult(T&);

  edm::EDGetTokenT<MTDTrackingRecHit> theMTDToken;

  // caches that should get filled once per event
  edm::Handle<edmNew::DetSetVector<MTDTrackingRecHit>> theMTDRecHits;

  void checkMTDRecHits();

  // keeps track of which event the cache holds
  edm::Event::CacheIdentifier_t theMTDEventCacheID;

  const edm::Event* theEvent;
};
#endif
