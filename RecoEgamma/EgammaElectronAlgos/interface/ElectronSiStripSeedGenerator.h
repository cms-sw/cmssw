#ifndef ElectronSiStripSeedGenerator_H
#define ElectronSiStripSeedGenerator_H

/** \class ElectronSiStripSeedGenerator
 
 * Class to generate the trajectory seed from two Si Strip hits.
 *  
 * \author Chris Macklin, Avishek Chatterjee 
 *
 * \version July 2008 
 *
 ************************************************************/
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
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
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/ElectronPixelSeedFwd.h"  
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"

#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"

#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

class PropagatorWithMaterial;
class KFUpdator;
class MeasurementTracker;
class NavigationSchool;

class ElectronSiStripSeedGenerator
{
public:

  //RC
  typedef edm::OwnVector<TrackingRecHit> PRecHitContainer;
  typedef TransientTrackingRecHit::ConstRecHitPointer  ConstRecHitPointer;
  typedef TransientTrackingRecHit::RecHitPointer       RecHitPointer;
  typedef TransientTrackingRecHit::RecHitContainer     RecHitContainer;
  

  enum mode{HLT, offline, unknown};  //to be used later

  ElectronSiStripSeedGenerator( );

  ~ElectronSiStripSeedGenerator();

  void setupES(const edm::EventSetup& setup, const edm::ParameterSet& conf);
  void run(edm::Event&, const edm::Handle<reco::SuperClusterCollection>&,
	   reco::ElectronPixelSeedCollection&);	

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
	
	void findSeedsFromCluster(edm::Ref<reco::SuperClusterCollection>,
				  reco::ElectronPixelSeedCollection&);
	
	int whichSubdetector(std::vector<const SiStripMatchedRecHit2D*>::const_iterator hit);

	bool preselection(GlobalPoint position,GlobalPoint superCluster,double phiVsRSlope);
	
	bool checkHitsAndTSOS(std::vector<const SiStripMatchedRecHit2D*>::const_iterator hit1,
			      std::vector<const SiStripMatchedRecHit2D*>::const_iterator hit2,
			      double scr,double scz,double pT,double scEta);

	const SiStripMatchedRecHit2D* matchedHitConverter(ConstRecHitPointer crhp);

	std::vector<bool> useDetLayer(double scEta);
	
	edm::ESHandle<MeasurementTracker> measurementTrackerHandle;
	edm::ESHandle<MagneticField> theMagField;
	edm::ESHandle<TrackerGeometry> trackerHandle;
	KFUpdator* theUpdator;
	PropagatorWithMaterial* thePropagator;	
	Chi2MeasurementEstimator* theEstimator;
	
	const NavigationSchool* theNavigationSchool;
	
	const edm::EventSetup *theSetup; 
	TrajectoryStateTransform transformer_; 
	PRecHitContainer recHits_; 
	PTrajectoryStateOnDet* pts_; 
	
	// member vectors to hold the good hits found between hit selection and combinatorics
	std::vector<const SiStripMatchedRecHit2D*> layer1Hits_;
	std::vector<const SiStripMatchedRecHit2D*> layer2Hits_;
	
	const SiStripRecHitMatcher* theMatcher_;

	std::string measurementTrackerName_;
	
};

#endif // ElectronSiStripSeedGenerator_H


