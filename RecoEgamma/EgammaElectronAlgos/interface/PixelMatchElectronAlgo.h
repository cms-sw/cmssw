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

#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SeedSuperClusterAssociation.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleaner.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"

#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
//#include "RecoTracker/CkfPattern/interface/CkfTrajectoryBuilder.h"
#include "RecoTracker/CkfPattern/interface/TrackerTrajectoryBuilder.h"
#include "RecoTracker/TkNavigation/interface/SimpleNavigationSchool.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/CkfPattern/interface/TransientInitialStateEstimator.h"
//CC@@
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"

#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/GlobalVector.h"

//class TransientInitialStateEstimator;

using namespace std;
using namespace edm;
using namespace reco;

class PixelMatchElectronAlgo {

public:

  PixelMatchElectronAlgo(double maxEOverP, double maxHOverE, 
                         double maxDeltaEta, double maxDeltaPhi);

  ~PixelMatchElectronAlgo();

  void setupES(const EventSetup& setup, const ParameterSet& conf);
  //  void run(const Event&, TrackCandidateCollection&, ElectronCollection&);
  void run(Event&, ElectronCollection&);

 private:

  // create electrons from tracks
  void process(edm::Handle<TrackCollection> tracksH, const SeedSuperClusterAssociationCollection *sclAss, ElectronCollection & outEle);
   // preselection method
  bool preSelection(const SuperCluster& clus, const Track& track);
  
  // temporary to get seed corresponding to track
  bool equal(edm::Ref<TrajectorySeedCollection> ts, const Track& t);
  bool compareHits(const TrackingRecHit& rh1, const TrackingRecHit & rh2) const ;

 // preselection parameters
  // maximum E/p where E is the supercluster corrected energy and p the track momentum at innermost state  
  double maxEOverP_;   
  // maximum H/E where H is the hadronic energy from the Hcal tower just behind the seed cluster and E the seed cluster energy  
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
  //  CkfTrajectoryBuilder*  theCkfTrajectoryBuilder;
  TrajectoryCleaner*               theTrajectoryCleaner;
  TransientInitialStateEstimator*  theInitialStateEstimator;
  
  ESHandle<MagneticField>                theMagField;
  ESHandle<GeometricSearchTracker>       theGeomSearchTracker;

  const MeasurementTracker*     theMeasurementTracker;
  const NavigationSchool*       theNavigationSchool;

};

#endif // PixelMatchElectronAlgo_H


