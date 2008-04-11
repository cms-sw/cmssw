#ifndef GsfElectronAlgo_H
#define GsfElectronAlgo_H

/** \class GsfElectronAlgo
 
 * Class to reconstruct electron tracks from electron pixel seeds
 *  keep track of information about the initiating supercluster
 *
 * \author U.Berthon, C.Charlot, LLR Palaiseau
 *
 * \version   2nd Version Oct 10, 2006  
 *
 ************************************************************/

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollections.h"

#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

class MultiTrajectoryStateTransform;
class GsfPropagatorAdapter;
 
class GsfElectronAlgo {

public:

  GsfElectronAlgo(const edm::ParameterSet& conf,
                         double maxEOverPBarrel, double maxEOverPEndcaps, 
                         double minEOverPBarrel, double minEOverPEndcaps,
                         double maxDeltaEta, double maxDeltaPhi, 
			 bool highPtPresel, double highPtMin,
		         bool applyEtaCorrection);
  ~GsfElectronAlgo();

  void setupES(const edm::EventSetup& setup);
  void run(edm::Event&, reco::GsfElectronCollection&);

 private:

  // create electrons from superclusters, tracks and Hcal rechits
  void process(edm::Handle<reco::GsfTrackCollection> tracksH,
	       const math::XYZPoint &bs,
	       reco::GsfElectronCollection & outEle);
  void process(edm::Handle<reco::GsfTrackCollection> tracksH,
	       edm::Handle<reco::SuperClusterCollection> superClustersBarrelH,
	       edm::Handle<reco::SuperClusterCollection> superClustersEndcapH,
	       const math::XYZPoint &bs,
	       reco::GsfElectronCollection & outEle);
  
  
  // preselection method
  bool preSelection(const reco::SuperCluster& clus);

  // interface to be improved...
  void createElectron(const reco::SuperClusterRef & scRef,
                      const reco::GsfTrackRef &trackRef,
		      reco::GsfElectronCollection & outEle);  

  //Gsf mode calculations
  GlobalVector computeMode(const TrajectoryStateOnSurface &tsos);

  // associations
  const reco::SuperClusterRef getTrSuperCluster(const reco::GsfTrackRef & trackRef);
  const reco::GsfTrackRef
    superClusterMatching(reco::SuperClusterRef sc, edm::Handle<reco::GsfTrackCollection> tracks);

  // intermediate calculations
  bool calculateTSOS(const reco::GsfTrack &t,const reco::SuperCluster & theClus, const math::XYZPoint & bs);

  // preselection parameters
  // maximum E/p where E is the supercluster corrected energy and p the track momentum at innermost state  
  double maxEOverPBarrel_;   
  double maxEOverPEndcaps_;   
  // minimum E/p where E is the supercluster corrected energy and p the track momentum at innermost state  
  double minEOverPBarrel_;   
  double minEOverPEndcaps_;     
  // maximum eta difference between the supercluster position and the track position at the closest impact to the supercluster 
  double maxDeltaEta_;
  // maximum phi difference between the supercluster position and the track position at the closest impact to the supercluster
  // position to the supercluster
  double maxDeltaPhi_;

  // high pt preselection parameters
  bool highPtPreselection_;
  double highPtMin_;
  
  //if this parameter is true the result of fEta correction will be set as energy of eletron
  bool applyEtaCorrection_;
  
  // input configuration
  edm::InputTag barrelSuperClusters_;
  edm::InputTag endcapSuperClusters_;
  edm::InputTag tracks_;
  edm::InputTag hcalRecHits_;

  edm::ESHandle<MagneticField>                theMagField;
  edm::ESHandle<CaloGeometry>                 theCaloGeom;
  edm::ESHandle<TrackerGeometry>              trackerHandle_;

  const MultiTrajectoryStateTransform *mtsTransform_;
  const GsfPropagatorAdapter *geomPropBw_;
  const GsfPropagatorAdapter *geomPropFw_;

  // internal variables 
  int subdet_; //subdetector for this cluster
  GlobalPoint sclPos_;
  GlobalVector vtxMom_;
  TrajectoryStateOnSurface innTSOS_;
  TrajectoryStateOnSurface outTSOS_;
  TrajectoryStateOnSurface vtxTSOS_;
  TrajectoryStateOnSurface sclTSOS_;
  TrajectoryStateOnSurface seedTSOS_;

  HBHERecHitMetaCollection *mhbhe_;
  unsigned int processType_;

  unsigned long long cacheIDGeom_;
  unsigned long long cacheIDTDGeom_;
  unsigned long long cacheIDMagField_;
};

#endif // GsfElectronAlgo_H


