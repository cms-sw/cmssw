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
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollections.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

class MultiTrajectoryStateTransform;
class GsfPropagatorAdapter;
 
class GsfElectronAlgo {

public:

  GsfElectronAlgo(const edm::ParameterSet& conf,
                         double maxEOverPBarrel, double maxEOverPEndcaps, 
                         double minEOverPBarrel, double minEOverPEndcaps,
                         double hOverEConeSize, double maxHOverE, 
                         double maxDeltaEta, double maxDeltaPhi, 
			 bool highPtPresel, double highPtMin,
		         bool applyEtaCorrection);
  ~GsfElectronAlgo();

  void setupES(const edm::EventSetup& setup);
  void run(edm::Event&, reco::GsfElectronCollection&);

 private:

  // create electrons from superclusters, tracks and Hcal rechits
  void process(edm::Handle<reco::GsfTrackCollection> tracksH,
	       const reco::BasicClusterShapeAssociationCollection *shpAssBarrel,
	       const reco::BasicClusterShapeAssociationCollection *shpAssEndcap,
	       HBHERecHitMetaCollection *mhbhe,
	       const math::XYZPoint &bs,
	       reco::GsfElectronCollection & outEle);
  void process(edm::Handle<reco::GsfTrackCollection> tracksH,
	       edm::Handle<reco::SuperClusterCollection> superClustersBarrelH,
	       edm::Handle<reco::SuperClusterCollection> superClustersEndcapH,
	       const reco::BasicClusterShapeAssociationCollection *shpAssBarrel,
	       const reco::BasicClusterShapeAssociationCollection *shpAssEndcap,
	       HBHERecHitMetaCollection *mhbhe,
	       const math::XYZPoint &bs,
	       reco::GsfElectronCollection & outEle);
  
  
  // preselection method
  //  bool preSelection(const reco::SuperCluster& clus, const GlobalVector&, const GlobalPoint&,double HoE);
  bool preSelection(const reco::SuperCluster& clus);

  // interface to be improved...
  void createElectron(const reco::SuperClusterRef & scRef,
                      const reco::GsfTrackRef &trackRef,const reco::ClusterShapeRef& seedShapeRef,
                      reco::GsfElectronCollection & outEle);  

  //Gsf mode calculations
  GlobalVector computeMode(const TrajectoryStateOnSurface &tsos);

  // associations
  const reco::SuperClusterRef getTrSuperCluster(const reco::GsfTrackRef & trackRef);
  const reco::GsfTrackRef
    superClusterMatching(reco::SuperClusterRef sc, edm::Handle<reco::GsfTrackCollection> tracks);

  // intermediate calculations
  void hOverE(const reco::SuperClusterRef & scRef,HBHERecHitMetaCollection *mhbhe);
  bool calculateTSOS(const reco::GsfTrack &t,const reco::SuperCluster & theClus,const math::XYZPoint & bs);

  //ecaleta, ecalphi: in fine to be replaced by propagators
  float ecalEta(float EtaParticle , float Zvertex, float plane_Radius);
  float ecalPhi(float PtParticle, float EtaParticle, float PhiParticle, int ChargeParticle, float Rstart);

  // preselection parameters
  // maximum E/p where E is the supercluster corrected energy and p the track momentum at innermost state  
  double maxEOverPBarrel_;   
  double maxEOverPEndcaps_;   
  // minimum E/p where E is the supercluster corrected energy and p the track momentum at innermost state  
  double minEOverPBarrel_;   
  double minEOverPEndcaps_;     
  // cone size for H/E
  double hOverEConeSize_; 
  // maximum H/E where H is the Hcal energy inside the cone centered on the seed cluster eta-phi position 
  double maxHOverE_; 
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
  std::string hbheLabel_;
  std::string hbheInstanceName_;
  std::string assBarrelShapeLabel_;
  std::string assBarrelShapeInstanceName_;
  std::string assEndcapShapeLabel_;
  std::string assEndcapShapeInstanceName_;
  std::string trackLabel_;
  std::string trackInstanceName_;

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
  double HoE_;
  TrajectoryStateOnSurface innTSOS_;
  TrajectoryStateOnSurface outTSOS_;
  TrajectoryStateOnSurface vtxTSOS_;
  TrajectoryStateOnSurface sclTSOS_;
  TrajectoryStateOnSurface seedTSOS_;

  unsigned int processType_;
};

#endif // GsfElectronAlgo_H


