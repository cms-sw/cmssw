#ifndef PixelMatchElectronAlgo_H
#define PixelMatchElectronAlgo_H

/** \class PixelMatchElectronAlgo
 
 * Class to reconstruct electron tracks from electron pixel seeds
 *  keep track of information about the initiating supercluster
 *
 * \author U.Berthon, C.Charlot, LLR Palaiseau
 *
 * \version   2nd Version Oct 10, 2006  
 *
 ************************************************************/

#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SeedSuperClusterAssociation.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollections.h"

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
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/GlobalVector.h"

class PixelMatchElectronAlgo {

public:

  PixelMatchElectronAlgo(double maxEOverPBarrel, double maxEOverPBarrel, 
                         double hOverEConeSize, double maxHOverE, 
                         double maxDeltaEta, double maxDeltaPhi);

  ~PixelMatchElectronAlgo();

  void setupES(const edm::EventSetup& setup, const edm::ParameterSet& conf);
  void run(edm::Event&, reco::PixelMatchGsfElectronCollection&);

 private:

  // create electrons from tracks
  void process(edm::Handle<reco::TrackCollection> tracksH, const reco::SeedSuperClusterAssociationCollection *sclAss,
   HBHERecHitMetaCollection mhbhe, reco::PixelMatchGsfElectronCollection & outEle);
  // preselection method
  bool preSelection(const reco::SuperCluster& clus, const reco::Track& track, HBHERecHitMetaCollection mhbhe);
  
  // temporary to get seed corresponding to track
  bool equal(edm::Ref<TrajectorySeedCollection> ts, const reco::Track& t);
  bool compareHits(const TrackingRecHit& rh1, const TrackingRecHit & rh2) const ;

 // preselection parameters
  // maximum E/p where E is the supercluster corrected energy and p the track momentum at innermost state  
  double maxEOverPBarrel_;   
  double maxEOverPEndcaps_;   
  // cone size for H/E
  double hOverEConeSize_; 
  // maximum H/E where H is the Hcal energy inside the cone centered on the seed cluster eta-phi position 
  double maxHOverE_; 
  // maximum eta difference between the supercluster position and the track position at the closest impact to the supercluster 
  double maxDeltaEta_;
  // maximum phi difference between the supercluster position and the track position at the closest impact to the supercluster
  // position to the supercluster
  double maxDeltaPhi_;
  
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
  TrajectoryCleaner*               theTrajectoryCleaner;
  TransientInitialStateEstimator*  theInitialStateEstimator;
  
  edm::ESHandle<MagneticField>                theMagField;
  edm::ESHandle<GeometricSearchTracker>       theGeomSearchTracker;
  edm::ESHandle<CaloGeometry>                 theCaloGeom;

  const MeasurementTracker*     theMeasurementTracker;
  const NavigationSchool*       theNavigationSchool;

};

#endif // PixelMatchElectronAlgo_H


