#ifndef SiStripElectronSeedGenerator_H
#define SiStripElectronSeedGenerator_H

/** \class SiStripElectronSeedGenerator

 * Class to generate the trajectory seed from two Si Strip hits.
 *
 * \author Chris Macklin, Avishek Chatterjee
 *
 * \version March 2009 (Adapt code to simplify call to SetupES)
 *
 ************************************************************/
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ValidityInterval.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h" 

#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"

#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

class PropagatorWithMaterial;
class KFUpdator;
class MeasurementTracker;
class MeasurementTrackerEvent;
class NavigationSchool;
class SiStripRecHitMatcher;

class SiStripElectronSeedGenerator
{
public:
  
  struct Tokens {
    edm::EDGetTokenT<reco::BeamSpot> token_bs;
    edm::EDGetTokenT<MeasurementTrackerEvent> token_mte;
  };

  typedef edm::OwnVector<TrackingRecHit> PRecHitContainer;
  typedef TransientTrackingRecHit::ConstRecHitPointer  ConstRecHitPointer;
  typedef TransientTrackingRecHit::RecHitPointer       RecHitPointer;
  typedef TransientTrackingRecHit::RecHitContainer     RecHitContainer;

  SiStripElectronSeedGenerator(const edm::ParameterSet&,
			       const Tokens&);

  ~SiStripElectronSeedGenerator();

  void setupES(const edm::EventSetup& setup);
  void run(edm::Event&, const edm::EventSetup& setup,
	   const edm::Handle<reco::SuperClusterCollection>&,
	   reco::ElectronSeedCollection&);

private:
  double normalPhi(double phi) const {
    while (phi > 2.* M_PI) { phi -= 2.*M_PI; }
    while (phi < 0) { phi += 2.*M_PI; }
    return phi;
  }

  double phiDiff(double phi1, double phi2){
    double result = normalPhi(phi1) - normalPhi(phi2);
    if(result > M_PI) result -= 2*M_PI;
    if(result < -M_PI) result += 2*M_PI;
    return result;
  }

  double unwrapPhi(double phi) const {
    while (phi > M_PI) { phi -= 2.*M_PI; }
    while (phi < -M_PI) { phi += 2.*M_PI; }
    return phi;
  }

  void findSeedsFromCluster(edm::Ref<reco::SuperClusterCollection>, edm::Handle<reco::BeamSpot>,
                            const MeasurementTrackerEvent &trackerData,
			    reco::ElectronSeedCollection&);

  int whichSubdetector(std::vector<const SiStripMatchedRecHit2D*>::const_iterator hit);

  bool preselection(GlobalPoint position,GlobalPoint superCluster,double phiVsRSlope, int hitLayer);
  //hitLayer: 1 = TIB, 2 = TID, 3 = TEC, 4 = Mono

  bool checkHitsAndTSOS(std::vector<const SiStripMatchedRecHit2D*>::const_iterator hit1,
			std::vector<const SiStripMatchedRecHit2D*>::const_iterator hit2,
			double scr,double scz,double pT,double scEta);

  bool altCheckHitsAndTSOS(std::vector<const SiStripMatchedRecHit2D*>::const_iterator hit1,
			   std::vector<const SiStripRecHit2D*>::const_iterator hit2,
			   double scr,double scz,double pT,double scEta);

  const SiStripMatchedRecHit2D* matchedHitConverter(ConstRecHitPointer crhp);
  const SiStripRecHit2D* backupHitConverter(ConstRecHitPointer crhp);

  std::vector<bool> useDetLayer(double scEta);

  edm::ESHandle<MeasurementTracker> measurementTrackerHandle;
  edm::ESHandle<MagneticField> theMagField;
  edm::ESHandle<TrackerGeometry> trackerGeometryHandle;
  edm::Handle<reco::BeamSpot> theBeamSpot;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotTag_;

  KFUpdator* theUpdator;
  PropagatorWithMaterial* thePropagator;
  Chi2MeasurementEstimator* theEstimator;

  std::string theMeasurementTrackerName;
  const MeasurementTracker* theMeasurementTracker;
  edm::EDGetTokenT<MeasurementTrackerEvent> theMeasurementTrackerEventTag;
  const edm::EventSetup *theSetup;
  
  PRecHitContainer recHits_;
  PTrajectoryStateOnDet pts_;

  // member vectors to hold the good hits found between hit selection and combinatorics
  std::vector<const SiStripMatchedRecHit2D*> layer1Hits_;
  std::vector<const SiStripMatchedRecHit2D*> layer2Hits_;
  std::vector<const SiStripRecHit2D*> backupLayer2Hits_;

  const SiStripRecHitMatcher* theMatcher_;

  unsigned long long cacheIDMagField_;
  unsigned long long cacheIDCkfComp_;
  unsigned long long cacheIDTrkGeom_;

  double tibOriginZCut_;
  double tidOriginZCut_;
  double tecOriginZCut_;
  double monoOriginZCut_;
  double tibDeltaPsiCut_;
  double tidDeltaPsiCut_;
  double tecDeltaPsiCut_;
  double monoDeltaPsiCut_;
  double tibPhiMissHit2Cut_;
  double tidPhiMissHit2Cut_;
  double tecPhiMissHit2Cut_;
  double monoPhiMissHit2Cut_;
  double tibZMissHit2Cut_;
  double tidRMissHit2Cut_;
  double tecRMissHit2Cut_;
  double tidEtaUsage_;
  int tidMaxHits_;
  int tecMaxHits_;
  int monoMaxHits_;
  int maxSeeds_;

};

#endif // SiStripElectronSeedGenerator_H


