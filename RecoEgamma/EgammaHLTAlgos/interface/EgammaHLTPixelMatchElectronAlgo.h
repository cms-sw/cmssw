#ifndef EgammaHLTPixelMatchElectronAlgo_H
#define EgammaHLTPixelMatchElectronAlgo_H

/** \class EgammaHLTPixelMatchElectronAlgo
 
 * Class to reconstruct electron tracks from electron pixel seeds
 *  keep track of information about the initiating supercluster
 *
 * \author Monica Vazquez Acosta (CERN)
 *
 ************************************************************/


#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"
#include "DataFormats/EgammaReco/interface/SeedSuperClusterAssociation.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleaner.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"

#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/CkfPattern/interface/TrackerTrajectoryBuilder.h"
#include "RecoTracker/TkNavigation/interface/SimpleNavigationSchool.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/CkfPattern/interface/TransientInitialStateEstimator.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"



//class TransientInitialStateEstimator;

using namespace std;
using namespace edm;
using namespace reco;

class EgammaHLTPixelMatchElectronAlgo {

public:

  EgammaHLTPixelMatchElectronAlgo(
				  //double maxEOverP, double maxHOverE, 
				  //double maxDeltaEta, double maxDeltaPhi
 );

  ~EgammaHLTPixelMatchElectronAlgo();

  void setupES(const EventSetup& setup, const ParameterSet& conf);
  //  void run(const Event&, TrackCandidateCollection&, ElectronCollection&);
  void run(Event&, ElectronCollection&);

 private:

  // create electrons from tracks
  void process(edm::Handle<TrackCollection> tracksH, const SeedSuperClusterAssociationCollection *sclAss, ElectronCollection & outEle);
  
  // temporary to get seed corresponding to track
  bool equal(edm::Ref<TrajectorySeedCollection> ts, const Track& t);
  bool compareHits(const TrackingRecHit& rh1, const TrackingRecHit & rh2) const ;

  // input configuration
  std::string trackBarrelLabel_;
  std::string trackEndcapLabel_;
  std::string trackBarrelInstanceName_;
  std::string trackEndcapInstanceName_;
  std::string assBarrelLabel_;
  std::string assBarrelInstanceName_;
  std::string assEndcapLabel_;
  std::string assEndcapInstanceName_;

  const TrackerTrajectoryBuilder*  theCkfTrajectoryBuilder;
  //  CkfTrajectoryBuilder*  theCkfTrajectoryBuilder;
  TrajectoryCleaner*               theTrajectoryCleaner;
  TransientInitialStateEstimator*  theInitialStateEstimator;
  
  ESHandle<MagneticField>                theMagField;
  ESHandle<GeometricSearchTracker>       theGeomSearchTracker;

  const MeasurementTracker*     theMeasurementTracker;
  const NavigationSchool*       theNavigationSchool;

};

#endif // EgammaHLTPixelMatchElectronAlgo_H


