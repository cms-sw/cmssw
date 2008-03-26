#ifndef MeasurementDet_MuonDetLayerMeasurements_H
#define MeasurementDet_MuonDetLayerMeasurements_H

/** \class MuonDetLayerMeasurements
 *  The class to access recHits and TrajectoryMeasurements from DetLayer.  
 *
 *  $Date: 2007/11/20 19:05:23 $
 *  $Revision: 1.15 $
 *  \author C. Liu, R. Bellan, N. Amapane
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
//#include "TrackingTools/ementDet/interface/TrajectoryMeasurement.h"
#include "TrackingTools/MeasurementDet/interface/TrajectoryMeasurementGroup.h"

#include "FWCore/ParameterSet/interface/InputTag.h"

#include <vector>

class DetLayer;
class GeomDet;
class TrajectoryMeasurement;


//FIXME: these typedefs MUST GO inside the scope of MuonDetLayerMeasurements
typedef std::vector<TrajectoryMeasurement>          MeasurementContainer;
typedef std::pair<const GeomDet*,TrajectoryStateOnSurface> DetWithState;


class MuonDetLayerMeasurements {
 public:


  MuonDetLayerMeasurements(edm::InputTag dtlabel,
			   edm::InputTag csclabel,
			   edm::InputTag rpclabel,
			   bool enableDT = true,
			   bool enableCSC = true,
			   bool enableRPC = true);
  
  virtual ~MuonDetLayerMeasurements();
  
  // for a given det and state.  Not clear when the fastMeasurements below
  //  should be used, since it isn't passed a GeomDet
  MeasurementContainer
    measurements( const DetLayer* layer,
                  const GeomDet * det,
                  const TrajectoryStateOnSurface& stateOnDet,
                  const MeasurementEstimator& est,
                  const edm::Event& iEvent) const;

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

  std::vector<TrajectoryMeasurementGroup>
    groupedMeasurements( const DetLayer* layer,
                  const TrajectoryStateOnSurface& startingState,
                  const Propagator& prop,
                  const MeasurementEstimator& est,
                  const edm::Event& iEvent) const;

  std::vector<TrajectoryMeasurementGroup>
    groupedMeasurements( const DetLayer* layer,
                  const TrajectoryStateOnSurface& startingState,
                  const Propagator& prop,
                  const MeasurementEstimator& est) const;
 
  void setEvent(const edm::Event &);  

  /// returns the rechits which are onto the layer
  MuonTransientTrackingRecHit::MuonRecHitContainer recHits(const DetLayer* layer, const edm::Event& iEvent) const;

  /// returns the rechits which are onto the layer
  MuonTransientTrackingRecHit::MuonRecHitContainer recHits(const DetLayer* layer) const;


 private:

  /// obtain TrackingRecHits from a DetLayer
  MuonTransientTrackingRecHit::MuonRecHitContainer recHits(const GeomDet*, const edm::Event& iEvent) const;

  /// check that the event is set, and throw otherwise
  void checkEvent() const;

  
  edm::InputTag theDTRecHitLabel;
  edm::InputTag theCSCRecHitLabel;
  edm::InputTag theRPCRecHitLabel;

  bool enableDTMeasurement;
  bool enableCSCMeasurement;
  bool enableRPCMeasurement;
  

  const edm::Event* theEvent;   
};
#endif

