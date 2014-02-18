#ifndef MeasurementDet_MuonDetLayerMeasurements_H
#define MeasurementDet_MuonDetLayerMeasurements_H

/** \class MuonDetLayerMeasurements
 *  The class to access recHits and TrajectoryMeasurements from DetLayer.  
 *
 *  \author C. Liu, R. Bellan, N. Amapane
 *  \modified by C. Calabria to include GEMs
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
//#include "TrackingTools/ementDet/interface/TrajectoryMeasurement.h"
#include "TrackingTools/MeasurementDet/interface/TrajectoryMeasurementGroup.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <vector>

class DetLayer;
class GeomDet;
class TrajectoryMeasurement;


//FIXME: these typedefs MUST GO inside the scope of MuonDetLayerMeasurements
typedef std::vector<TrajectoryMeasurement>          MeasurementContainer;
typedef std::pair<const GeomDet*,TrajectoryStateOnSurface> DetWithState;


class MuonDetLayerMeasurements {
 public:
  typedef MuonTransientTrackingRecHit::MuonRecHitContainer MuonRecHitContainer;

  MuonDetLayerMeasurements(edm::InputTag dtlabel,
			   edm::InputTag csclabel,
			   edm::InputTag rpclabel,
			   edm::InputTag gemlabel,
			   bool enableDT = true,
			   bool enableCSC = true,
			   bool enableRPC = true,
			   bool enableGEM = true);
  
  virtual ~MuonDetLayerMeasurements();
  
  // for a given det and state.  Not clear when the fastMeasurements below
  //  should be used, since it isn't passed a GeomDet
  MeasurementContainer
    measurements( const DetLayer* layer,
                  const GeomDet * det,
                  const TrajectoryStateOnSurface& stateOnDet,
                  const MeasurementEstimator& est,
                  const edm::Event& iEvent);

  /// returns TMeasurements in a DetLayer compatible with the TSOS.
  MeasurementContainer
    measurements( const DetLayer* layer,
		  const TrajectoryStateOnSurface& startingState,
		  const Propagator& prop,
		  const MeasurementEstimator& est,
		  const edm::Event& iEvent);

  /// faster version in case the TrajectoryState on the surface of the GeomDet is already available
  MeasurementContainer
    fastMeasurements( const DetLayer* layer,
		      const TrajectoryStateOnSurface& theStateOnDet,
		      const TrajectoryStateOnSurface& startingState,
		      const Propagator& prop,
		      const MeasurementEstimator& est,
		      const edm::Event& iEvent);

  /// returns TMeasurements in a DetLayer compatible with the TSOS.
  MeasurementContainer
    measurements( const DetLayer* layer,
		  const TrajectoryStateOnSurface& startingState,
		  const Propagator& prop,
		  const MeasurementEstimator& est);

  /// faster version in case the TrajectoryState on the surface of the GeomDet is already available
  MeasurementContainer
    fastMeasurements( const DetLayer* layer,
		      const TrajectoryStateOnSurface& theStateOnDet,
		      const TrajectoryStateOnSurface& startingState,
		      const Propagator& prop,
		      const MeasurementEstimator& est);

  std::vector<TrajectoryMeasurementGroup>
    groupedMeasurements( const DetLayer* layer,
                  const TrajectoryStateOnSurface& startingState,
                  const Propagator& prop,
                  const MeasurementEstimator& est,
                  const edm::Event& iEvent);

  std::vector<TrajectoryMeasurementGroup>
    groupedMeasurements( const DetLayer* layer,
                  const TrajectoryStateOnSurface& startingState,
                  const Propagator& prop,
                  const MeasurementEstimator& est);
 
  void setEvent(const edm::Event &);  

  /// returns the rechits which are on the layer
  MuonRecHitContainer recHits(const DetLayer* layer, const edm::Event& iEvent);

  /// returns the rechits which are on the layer
  MuonRecHitContainer recHits(const DetLayer* layer);


 private:

  /// obtain TrackingRecHits from a DetLayer
  MuonRecHitContainer recHits(const GeomDet*, const edm::Event& iEvent);

  /// check that the event is set, and throw otherwise
  void checkEvent() const;

  edm::InputTag theDTRecHitLabel;
  edm::InputTag theCSCRecHitLabel;
  edm::InputTag theRPCRecHitLabel;
  edm::InputTag theGEMRecHitLabel;

  bool enableDTMeasurement;
  bool enableCSCMeasurement;
  bool enableRPCMeasurement;
  bool enableGEMMeasurement;
  
  // caches that should get filled once per event
  edm::Handle<DTRecSegment4DCollection> theDTRecHits;
  edm::Handle<CSCSegmentCollection>     theCSCRecHits;
  edm::Handle<RPCRecHitCollection>      theRPCRecHits;
  edm::Handle<GEMRecHitCollection>      theGEMRecHits;

  void checkDTRecHits();
  void checkCSCRecHits();
  void checkRPCRecHits();
  void checkGEMRecHits();

  // keeps track of which event the cache holds
  edm::EventID theDTEventID;
  edm::EventID theCSCEventID;
  edm::EventID theRPCEventID;
  edm::EventID theGEMEventID;

  const edm::Event* theEvent;   

  // strings to uniquely identify current process
  std::string theDTCheckName;
  std::string theRPCCheckName;
  std::string theCSCCheckName;
  std::string theGEMCheckName;
};
#endif

