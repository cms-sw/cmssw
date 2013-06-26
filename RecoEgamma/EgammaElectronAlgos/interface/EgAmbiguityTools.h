
#ifndef EgAmbiguityTools_H
#define EgAmbiguityTools_H

class MultiTrajectoryStateTransform ;
class MultiTrajectoryStateMode ;

#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaRecHitIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/ElectronTkIsolation.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include <list>

namespace EgAmbiguityTools
 {
  // for clusters
  float sharedEnergy
   ( const reco::CaloCluster *, const reco::CaloCluster *,
	   edm::Handle<EcalRecHitCollection> & barrelRecHits,
	   edm::Handle<EcalRecHitCollection> & endcapRecHits ) ;
  float sharedEnergy
   ( const reco::SuperClusterRef &, const reco::SuperClusterRef &,
	   edm::Handle<EcalRecHitCollection> & barrelRecHits,
	   edm::Handle<EcalRecHitCollection> & endcapRecHits ) ;

  // for tracks
  int sharedHits( const reco::GsfTrackRef &, const reco::GsfTrackRef & ) ;
  int sharedDets( const reco::GsfTrackRef &, const reco::GsfTrackRef & ) ;

  // electrons comparison
  bool isBetter( const reco::GsfElectron *, const reco::GsfElectron * ) ;
  struct isInnerMost
   {
  	edm::ESHandle<TrackerGeometry> trackerHandle_ ;
	isInnerMost( edm::ESHandle<TrackerGeometry> & geom ) : trackerHandle_(geom) {}
	bool operator()( const reco::GsfElectron *, const reco::GsfElectron * ) ;
   } ;

 }

#endif


