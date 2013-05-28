#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"

//COLLECTION
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"


///ESHANDLE
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"


class DetLayer;
class TrajectoryStateOnSurface;
class ParticlePropagator;
class TrackerLayer;
class MagneticField;
class TrackerInteractionGeometry;
class TrackerGeometry;
class MagneticFieldMap;
class PropagatorWithMaterial;
class KFUpdator;
class TransientTrackingRecHitBuilder;
class TrajectoryStateTransform;

class ConvBremSeedProducer : public edm::EDProducer {
  typedef SiStripRecHit2DCollection::const_iterator StDetMatch;
  typedef SiPixelRecHitCollection::const_iterator PiDetMatch;
  typedef SiStripMatchedRecHit2DCollection::const_iterator MatDetMatch;
  typedef SiStripRecHit2DCollection::DetSet        StDetSet;
  typedef SiPixelRecHitCollection::DetSet          PiDetSet;
  typedef SiStripMatchedRecHit2DCollection::DetSet MatDetSet;
  typedef GeometricSearchDet::DetWithState   DetWithState;
 
 public:
  explicit ConvBremSeedProducer(const edm::ParameterSet&);
  ~ConvBremSeedProducer();
  
 private:
  virtual void beginRun(const edm::Run&,const edm::EventSetup&) override;
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endRun(const edm::Run&,const edm::EventSetup&) override;
  void initializeLayerMap();
  std::vector<const DetLayer*>                theLayerMap;
  TrajectoryStateOnSurface makeTrajectoryState( const DetLayer* layer, 
						const ParticlePropagator& pp,
						const MagneticField* field) const;
  const DetLayer* detLayer( const TrackerLayer& layer, float zpos) const;

  bool isGsfTrack(const TrackingRecHitRefVector&, const TrackingRecHit *);

  int GoodCluster(const BaseParticlePropagator& bpg, const reco::PFClusterCollection& pfc, 
		  float minep, bool sec=false);

  std::vector <bool> sharedHits( const std::vector<std::pair< TrajectorySeed, 
				 std::pair<GlobalVector,float> > >& );

  edm::ParameterSet                           conf_;
  const GeometricSearchTracker*               geomSearchTracker_;
  const TrackerInteractionGeometry*           geometry_;
  const TrackerGeometry*                      tracker_;
  const MagneticField*                        magfield_;
  const MagneticFieldMap*                     fieldMap_;
  const PropagatorWithMaterial*               propagator_;
  const KFUpdator*                            kfUpdator_;
  const TransientTrackingRecHitBuilder*       hitBuilder_;
  const TrajectoryStateTransform*             transformer_;
  std::vector<const DetLayer*>                layerMap_;
  int                                         negLayerOffset_;
  ///B field
  math::XYZVector B_;

};
