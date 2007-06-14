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

#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SeedSuperClusterAssociation.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/PixelMatchTrackReco/interface/GsfTrackSeedAssociation.h"

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
 
class PixelMatchElectronAlgo {

public:

  PixelMatchElectronAlgo(double maxEOverPBarrel, double maxEOverPEndcaps, 
                         double minEOverPBarrel, double minEOverPEndcaps,
                         double hOverEConeSize, double maxHOverE, 
                         double maxDeltaEta, double maxDeltaPhi, double ptCut);

  ~PixelMatchElectronAlgo();

  void setupES(const edm::EventSetup& setup, const edm::ParameterSet& conf);
  void run(edm::Event&, reco::PixelMatchGsfElectronCollection&);

 private:

  // create electrons from tracks
  void process(edm::Handle<reco::GsfTrackCollection> tracksH, const reco::SeedSuperClusterAssociationCollection *sclAss, const reco::GsfTrackSeedAssociationCollection *tsAss,
   HBHERecHitMetaCollection *mhbhe, reco::PixelMatchGsfElectronCollection & outEle);
  // preselection method
  //  bool preSelection(const reco::SuperCluster& clus, const reco::GsfTrack& track,double HoE);
 bool preSelection(const reco::SuperCluster& clus, const GlobalVector&, const GlobalPoint&,double HoE);
  

  //Gsf mode calculations
  GlobalVector computeMode(const TrajectoryStateOnSurface &tsos);

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

  double ptCut_;
 
  // input configuration
  std::string hbheLabel_;
  std::string hbheInstanceName_;
  std::string trackBarrelLabel_;
  std::string trackEndcapLabel_;
  std::string trackBarrelInstanceName_;
  std::string trackEndcapInstanceName_;
  std::string assBarrelLabel_;
  std::string assBarrelInstanceName_;
  std::string assEndcapLabel_;
  std::string assEndcapInstanceName_;
  std::string assBarrelTrTSLabel_;
  std::string assBarrelTrTSInstanceName_;
  std::string assEndcapTrTSLabel_;
  std::string assEndcapTrTSInstanceName_;

  edm::ESHandle<MagneticField>                theMagField;
  edm::ESHandle<CaloGeometry>                 theCaloGeom;
  edm::ESHandle<TrackerGeometry>              trackerHandle_;

  const MultiTrajectoryStateTransform *mtsTransform_;
  const GsfPropagatorAdapter *geomPropBw_;
  const GsfPropagatorAdapter *geomPropFw_;
};

#endif // PixelMatchElectronAlgo_H


