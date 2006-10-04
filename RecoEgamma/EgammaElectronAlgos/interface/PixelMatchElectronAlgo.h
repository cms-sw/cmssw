#ifndef PixelMatchElectronAlgo_H
#define PixelMatchElectronAlgo_H

/** \class PixelMatchElectronAlgo
 
 * Class to reconstruct electron tracks from electron pixel seeds
 *  keep track of information about the initiating supercluster
 *
 * \author U.Berthon, C.Charlot, LLR Palaiseau
 *
 * \version   1st Version July 6, 2006  
 *
 ************************************************************/

#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleaner.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"

#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/CkfPattern/interface/CkfTrajectoryBuilder.h"
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

  // preselection method
  bool preSelection(const SuperCluster& clus, const Track& track);
  
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
  
  string inputDataModuleLabel_;
  string inputDataInstanceName_;

  CkfTrajectoryBuilder*  theCkfTrajectoryBuilder;
  TrajectoryCleaner*               theTrajectoryCleaner;
  TransientInitialStateEstimator*  theInitialStateEstimator;
  
  ESHandle<MagneticField>                theMagField;
  ESHandle<GeometricSearchTracker>       theGeomSearchTracker;

  const MeasurementTracker*     theMeasurementTracker;
  const NavigationSchool*       theNavigationSchool;

};

#endif // PixelMatchElectronAlgo_H


