// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      SiStripElectronSeedGenerator.
//
/**\class SiStripElectronSeedGenerator EgammaElectronAlgos/SiStripElectronSeedGenerator

Description: SiStrip-driven electron seed finding algorithm.

*/
//
//

#include <vector>
#include <utility>

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/EgammaReco/interface/ElectronSeed.h"



#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripMatchedRecHit.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

// files for retrieving hits using measurement tracker

#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"


#include "RecoEgamma/EgammaElectronAlgos/interface/SiStripElectronSeedGenerator.h"

SiStripElectronSeedGenerator::SiStripElectronSeedGenerator(const edm::ParameterSet &pset, const SiStripElectronSeedGenerator::Tokens& tokens)
 : beamSpotTag_(tokens.token_bs),
   theUpdator(0),thePropagator(0),theMeasurementTracker(0),
   theMeasurementTrackerEventTag(tokens.token_mte),
   theSetup(0), theMatcher_(0),
   cacheIDMagField_(0),cacheIDCkfComp_(0),cacheIDTrkGeom_(0),
   tibOriginZCut_(pset.getParameter<double>("tibOriginZCut")),
   tidOriginZCut_(pset.getParameter<double>("tidOriginZCut")),
   tecOriginZCut_(pset.getParameter<double>("tecOriginZCut")),
   monoOriginZCut_(pset.getParameter<double>("monoOriginZCut")),
   tibDeltaPsiCut_(pset.getParameter<double>("tibDeltaPsiCut")),
   tidDeltaPsiCut_(pset.getParameter<double>("tidDeltaPsiCut")),
   tecDeltaPsiCut_(pset.getParameter<double>("tecDeltaPsiCut")),
   monoDeltaPsiCut_(pset.getParameter<double>("monoDeltaPsiCut")),
   tibPhiMissHit2Cut_(pset.getParameter<double>("tibPhiMissHit2Cut")),
   tidPhiMissHit2Cut_(pset.getParameter<double>("tidPhiMissHit2Cut")),
   tecPhiMissHit2Cut_(pset.getParameter<double>("tecPhiMissHit2Cut")),
   monoPhiMissHit2Cut_(pset.getParameter<double>("monoPhiMissHit2Cut")),
   tibZMissHit2Cut_(pset.getParameter<double>("tibZMissHit2Cut")),
   tidRMissHit2Cut_(pset.getParameter<double>("tidRMissHit2Cut")),
   tecRMissHit2Cut_(pset.getParameter<double>("tecRMissHit2Cut")),
   tidEtaUsage_(pset.getParameter<double>("tidEtaUsage")),
   tidMaxHits_(pset.getParameter<int>("tidMaxHits")),
   tecMaxHits_(pset.getParameter<int>("tecMaxHits")),
   monoMaxHits_(pset.getParameter<int>("monoMaxHits")),
   maxSeeds_(pset.getParameter<int>("maxSeeds"))
{
  // use of a theMeasurementTrackerName
  if (pset.exists("measurementTrackerName"))
   { theMeasurementTrackerName = pset.getParameter<std::string>("measurementTrackerName") ; }
  

  // new beamSpot tag
  /*
  if (pset.exists("beamSpot"))
   { beamSpotTag_ = pset.getParameter<edm::InputTag>("beamSpot") ; }
  */

  theUpdator = new KFUpdator();
  theEstimator = new Chi2MeasurementEstimator(30,3);
}


SiStripElectronSeedGenerator::~SiStripElectronSeedGenerator() {
  delete thePropagator;
  delete theUpdator;
}


void SiStripElectronSeedGenerator::setupES(const edm::EventSetup& setup) {

  if (cacheIDMagField_!=setup.get<IdealMagneticFieldRecord>().cacheIdentifier()) {
    setup.get<IdealMagneticFieldRecord>().get(theMagField);
    cacheIDMagField_=setup.get<IdealMagneticFieldRecord>().cacheIdentifier();
    if (thePropagator) delete thePropagator;
    thePropagator = new PropagatorWithMaterial(alongMomentum,.000511,&(*theMagField));
  }

  if (cacheIDCkfComp_!=setup.get<CkfComponentsRecord>().cacheIdentifier()) {
    setup.get<CkfComponentsRecord>().get(theMeasurementTrackerName,measurementTrackerHandle);
    cacheIDCkfComp_=setup.get<CkfComponentsRecord>().cacheIdentifier();
    theMeasurementTracker = measurementTrackerHandle.product();
  }

  if (cacheIDTrkGeom_!=setup.get<TrackerDigiGeometryRecord>().cacheIdentifier()) {
    setup.get<TrackerDigiGeometryRecord>().get(trackerGeometryHandle);
    cacheIDTrkGeom_=setup.get<TrackerDigiGeometryRecord>().cacheIdentifier();
  }

}

void  SiStripElectronSeedGenerator::run(edm::Event& e, const edm::EventSetup& setup,
					const edm::Handle<reco::SuperClusterCollection> &clusters,
					reco::ElectronSeedCollection & out) {
  theSetup= &setup;

  e.getByToken(beamSpotTag_,theBeamSpot);
  edm::Handle<MeasurementTrackerEvent> data;
  e.getByToken(theMeasurementTrackerEventTag, data);


  for  (unsigned int i=0;i<clusters->size();++i) {
    edm::Ref<reco::SuperClusterCollection> theClusB(clusters,i);
    // Find the seeds
    LogDebug ("run") << "new cluster, calling findSeedsFromCluster";
    findSeedsFromCluster(theClusB,theBeamSpot,*data,out);
  }

  LogDebug ("run") << ": For event "<<e.id();
  LogDebug ("run") <<"Nr of superclusters: "<<clusters->size()
		   <<", no. of ElectronSeeds found  = " << out.size();
}


// Find seeds using a supercluster
void SiStripElectronSeedGenerator::findSeedsFromCluster
 ( edm::Ref<reco::SuperClusterCollection> seedCluster,
   edm::Handle<reco::BeamSpot> bs,
   const MeasurementTrackerEvent & trackerData,
	 reco::ElectronSeedCollection & result )
 {
  // clear the member vectors of good hits
  layer1Hits_.clear() ;
  layer2Hits_.clear() ;
  backupLayer2Hits_.clear() ;

  using namespace std;

  double sCenergy = seedCluster->energy();
  math::XYZPoint sCposition = seedCluster->position();
  double scEta = seedCluster->eta();

  double scz = sCposition.z();
  double scr = sqrt(pow(sCposition.x(),2)+pow(sCposition.y(),2));

  double pT = sCenergy * seedCluster->position().rho()/sqrt(seedCluster->x()*seedCluster->x()+seedCluster->y()*seedCluster->y()+seedCluster->z()*seedCluster->z());

  // FIXME use nominal field (see below)
  double magneticField = 3.8;

  // cf Jackson p. 581-2, a little geometry
  double phiVsRSlope = -3.00e-3 * magneticField / pT / 2.;


  //Need to create TSOS to feed MeasurementTracker
  GlobalPoint beamSpot(bs->x0(),bs->y0(),bs->z0());
  GlobalPoint superCluster(sCposition.x(),sCposition.y(),sCposition.z());
  double r0 = beamSpot.perp();
  double z0 = beamSpot.z();

  //We need to pick a charge for the particle we want to reconstruct before hits can be retrieved
  //Choosing both charges improves seeding efficiency by less than 0.5% for signal events
  //If we pick a single charge, this reduces fake rate and CPU time
  //So we pick a charge that is equally likely to be positive or negative

  int chargeHypothesis;
  double chargeSelector = sCenergy - (int)sCenergy;
  if(chargeSelector >= 0.5) chargeHypothesis = -1;
  if(chargeSelector < 0.5) chargeHypothesis = 1;

  //Use BeamSpot and SC position to estimate 3rd point
  double rFake = 25.;
  double phiFake = phiDiff(superCluster.phi(),chargeHypothesis * phiVsRSlope * (scr - rFake));
  double zFake = (rFake*(scz-z0)-r0*scz+scr*z0)/(scr-r0);
  double xFake = rFake * cos(phiFake);
  double yFake = rFake * sin(phiFake);
  GlobalPoint fakePoint(xFake,yFake,zFake);

  // FIXME optmize, move outside loop
  edm::ESHandle<MagneticField> bfield;
  theSetup->get<IdealMagneticFieldRecord>().get(bfield);
  float nomField = bfield->nominalValue();

  //Use 3 points to make helix
  FastHelix initialHelix(superCluster,fakePoint,beamSpot,nomField,&*bfield);

  //Use helix to get FTS
  FreeTrajectoryState initialFTS(initialHelix.stateAtVertex());

  //Use FTS and BeamSpot to create TSOS
  TransverseImpactPointExtrapolator* tipe = new TransverseImpactPointExtrapolator(*thePropagator);
  TrajectoryStateOnSurface initialTSOS = tipe->extrapolate(initialFTS,beamSpot);

  //Use GST to retrieve hits from various DetLayers using layerMeasurements class
  const GeometricSearchTracker* gst = theMeasurementTracker->geometricSearchTracker();

  std::vector<BarrelDetLayer*> tibLayers = gst->tibLayers();
  DetLayer* tib1 = tibLayers.at(0);
  DetLayer* tib2 = tibLayers.at(1);

  std::vector<ForwardDetLayer*> tecLayers;
  std::vector<ForwardDetLayer*> tidLayers;
  if(scEta < 0){
    tecLayers = gst->negTecLayers();
    tidLayers = gst->negTidLayers();
  }
  if(scEta > 0){
    tecLayers = gst->posTecLayers();
    tidLayers = gst->posTidLayers();
  }

  DetLayer* tid1 = tidLayers.at(0);
  DetLayer* tid2 = tidLayers.at(1);
  DetLayer* tid3 = tidLayers.at(2);
  DetLayer* tec1 = tecLayers.at(0);
  DetLayer* tec2 = tecLayers.at(1);
  DetLayer* tec3 = tecLayers.at(2);

  //Figure out which DetLayers to use based on SC Eta
  std::vector<bool> useDL = useDetLayer(scEta);
  bool useTID = false;

  //Use counters to restrict the number of hits in TID and TEC layers
  //This reduces seed multiplicity
  int tid1MHC = 0;
  int tid2MHC = 0;
  int tid3MHC = 0;
  int tid1BHC = 0;
  int tid2BHC = 0;
  int tid3BHC = 0;
  int tec1MHC = 0;
  int tec2MHC = 0;
  int tec3MHC = 0;

  //Use counter to limit the allowed number of seeds
  int seedCounter = 0;

  bool hasLay1Hit = false;
  bool hasLay2Hit = false;
  bool hasBackupHit = false;

  LayerMeasurements layerMeasurements(*theMeasurementTracker, trackerData);

  std::vector<TrajectoryMeasurement> tib1measurements;
  if(useDL.at(0)) tib1measurements = layerMeasurements.measurements(*tib1,initialTSOS,*thePropagator,*theEstimator);
  std::vector<TrajectoryMeasurement> tib2measurements;
  if(useDL.at(1)) tib2measurements = layerMeasurements.measurements(*tib2,initialTSOS,*thePropagator,*theEstimator);

  //Basic idea: Retrieve hits from a given DetLayer
  //Check if it is a Matched Hit and satisfies some cuts
  //If yes, accept hit for seed making

  for(std::vector<TrajectoryMeasurement>::const_iterator tmIter = tib1measurements.begin(); tmIter != tib1measurements.end(); ++ tmIter){
    ConstRecHitPointer hit = tmIter->recHit();
    const SiStripMatchedRecHit2D* matchedHit = matchedHitConverter(hit);
    if(matchedHit){
      GlobalPoint position = trackerGeometryHandle->idToDet(matchedHit->geographicalId())->surface().toGlobal(matchedHit->localPosition());
      if(preselection(position, superCluster, phiVsRSlope, 1)){
	hasLay1Hit = true;
	layer1Hits_.push_back(matchedHit);
      }
    }
  }

  for(std::vector<TrajectoryMeasurement>::const_iterator tmIter = tib2measurements.begin(); tmIter != tib2measurements.end(); ++ tmIter){
    ConstRecHitPointer hit = tmIter->recHit();
    const SiStripMatchedRecHit2D* matchedHit = matchedHitConverter(hit);
    if(matchedHit){
      GlobalPoint position = trackerGeometryHandle->idToDet(matchedHit->geographicalId())->surface().toGlobal(matchedHit->localPosition());
      if(preselection(position, superCluster, phiVsRSlope, 1)){
	hasLay2Hit = true;
	layer2Hits_.push_back(matchedHit);
      }
    }
  }

  if(!(hasLay1Hit && hasLay2Hit)) useTID = true;
  if(std::abs(scEta) > tidEtaUsage_) useTID = true;
  std::vector<TrajectoryMeasurement> tid1measurements;
  if(useDL.at(2) && useTID) tid1measurements = layerMeasurements.measurements(*tid1,initialTSOS,*thePropagator,*theEstimator);
  std::vector<TrajectoryMeasurement> tid2measurements;
  if(useDL.at(3) && useTID) tid2measurements = layerMeasurements.measurements(*tid2,initialTSOS,*thePropagator,*theEstimator);
  std::vector<TrajectoryMeasurement> tid3measurements;
  if(useDL.at(4) && useTID) tid3measurements = layerMeasurements.measurements(*tid3,initialTSOS,*thePropagator,*theEstimator);

  for(std::vector<TrajectoryMeasurement>::const_iterator tmIter = tid1measurements.begin(); tmIter != tid1measurements.end(); ++ tmIter){
    if(tid1MHC < tidMaxHits_){
    ConstRecHitPointer hit = tmIter->recHit();
    const SiStripMatchedRecHit2D* matchedHit = matchedHitConverter(hit);
    if(matchedHit){
      GlobalPoint position = trackerGeometryHandle->idToDet(matchedHit->geographicalId())->surface().toGlobal(matchedHit->localPosition());
      if(preselection(position, superCluster, phiVsRSlope, 2)){
	tid1MHC++;
	hasLay1Hit = true;
	layer1Hits_.push_back(matchedHit);
	hasLay2Hit = true;
	layer2Hits_.push_back(matchedHit);
      }
    }else if(useDL.at(8) && tid1BHC < monoMaxHits_){
      const SiStripRecHit2D* backupHit = backupHitConverter(hit);
      if(backupHit){
	GlobalPoint position = trackerGeometryHandle->idToDet(backupHit->geographicalId())->surface().toGlobal(backupHit->localPosition());
	if(preselection(position, superCluster, phiVsRSlope, 4) && position.perp() > 37.){
	  tid1BHC++;
	  hasBackupHit = true;
	  backupLayer2Hits_.push_back(backupHit);
	}
      }
    }
    }
  }

  for(std::vector<TrajectoryMeasurement>::const_iterator tmIter = tid2measurements.begin(); tmIter != tid2measurements.end(); ++ tmIter){
    if(tid2MHC < tidMaxHits_){
    ConstRecHitPointer hit = tmIter->recHit();
    const SiStripMatchedRecHit2D* matchedHit = matchedHitConverter(hit);
    if(matchedHit){
      GlobalPoint position = trackerGeometryHandle->idToDet(matchedHit->geographicalId())->surface().toGlobal(matchedHit->localPosition());
      if(preselection(position, superCluster, phiVsRSlope, 2)){
	tid2MHC++;
	hasLay1Hit = true;
	layer1Hits_.push_back(matchedHit);
	hasLay2Hit = true;
	layer2Hits_.push_back(matchedHit);
      }
    }else if(useDL.at(8) && tid2BHC < monoMaxHits_){
      const SiStripRecHit2D* backupHit = backupHitConverter(hit);
      if(backupHit){
	GlobalPoint position = trackerGeometryHandle->idToDet(backupHit->geographicalId())->surface().toGlobal(backupHit->localPosition());
	if(preselection(position, superCluster, phiVsRSlope, 4) && position.perp() > 37.){
	  tid2BHC++;
	  hasBackupHit = true;
	  backupLayer2Hits_.push_back(backupHit);
	}
      }
    }
    }
  }

  for(std::vector<TrajectoryMeasurement>::const_iterator tmIter = tid3measurements.begin(); tmIter != tid3measurements.end(); ++ tmIter){
    if(tid3MHC < tidMaxHits_){
    ConstRecHitPointer hit = tmIter->recHit();
    const SiStripMatchedRecHit2D* matchedHit = matchedHitConverter(hit);
    if(matchedHit){
      GlobalPoint position = trackerGeometryHandle->idToDet(matchedHit->geographicalId())->surface().toGlobal(matchedHit->localPosition());
      if(preselection(position, superCluster, phiVsRSlope, 2)){
	tid3MHC++;
	hasLay1Hit = true;
	layer1Hits_.push_back(matchedHit);
	hasLay2Hit = true;
	layer2Hits_.push_back(matchedHit);
      }
    }else if(useDL.at(8) && tid3BHC < monoMaxHits_){
      const SiStripRecHit2D* backupHit = backupHitConverter(hit);
      if(backupHit){
	GlobalPoint position = trackerGeometryHandle->idToDet(backupHit->geographicalId())->surface().toGlobal(backupHit->localPosition());
	if(preselection(position, superCluster, phiVsRSlope, 4) && position.perp() > 37.){
	  tid3BHC++;
	  hasBackupHit = true;
	  backupLayer2Hits_.push_back(backupHit);
	}
      }
    }
    }
  }

  std::vector<TrajectoryMeasurement> tec1measurements;
  if(useDL.at(5)) tec1measurements = layerMeasurements.measurements(*tec1,initialTSOS,*thePropagator,*theEstimator);
  std::vector<TrajectoryMeasurement> tec2measurements;
  if(useDL.at(6)) tec2measurements = layerMeasurements.measurements(*tec2,initialTSOS,*thePropagator,*theEstimator);
  std::vector<TrajectoryMeasurement> tec3measurements;
  if(useDL.at(7)) tec3measurements = layerMeasurements.measurements(*tec3,initialTSOS,*thePropagator,*theEstimator);

  for(std::vector<TrajectoryMeasurement>::const_iterator tmIter = tec1measurements.begin(); tmIter != tec1measurements.end(); ++ tmIter){
    if(tec1MHC < tecMaxHits_){
    ConstRecHitPointer hit = tmIter->recHit();
    const SiStripMatchedRecHit2D* matchedHit = matchedHitConverter(hit);
    if(matchedHit){
      GlobalPoint position = trackerGeometryHandle->idToDet(matchedHit->geographicalId())->surface().toGlobal(matchedHit->localPosition());
      if(preselection(position, superCluster, phiVsRSlope, 3)){
	tec1MHC++;
	hasLay1Hit = true;
	layer1Hits_.push_back(matchedHit);
	hasLay2Hit = true;
	layer2Hits_.push_back(matchedHit);
      }
    }
    }
  }

  for(std::vector<TrajectoryMeasurement>::const_iterator tmIter = tec2measurements.begin(); tmIter != tec2measurements.end(); ++ tmIter){
    if(tec2MHC < tecMaxHits_){
    ConstRecHitPointer hit = tmIter->recHit();
    const SiStripMatchedRecHit2D* matchedHit = matchedHitConverter(hit);
    if(matchedHit){
      GlobalPoint position = trackerGeometryHandle->idToDet(matchedHit->geographicalId())->surface().toGlobal(matchedHit->localPosition());
      if(preselection(position, superCluster, phiVsRSlope, 3)){
	tec2MHC++;
	hasLay1Hit = true;
	layer1Hits_.push_back(matchedHit);
	hasLay2Hit = true;
	layer2Hits_.push_back(matchedHit);
      }
    }
    }
  }

  for(std::vector<TrajectoryMeasurement>::const_iterator tmIter = tec3measurements.begin(); tmIter != tec3measurements.end(); ++ tmIter){
    if(tec3MHC < tecMaxHits_){
    ConstRecHitPointer hit = tmIter->recHit();
    const SiStripMatchedRecHit2D* matchedHit = matchedHitConverter(hit);
    if(matchedHit){
      GlobalPoint position = trackerGeometryHandle->idToDet(matchedHit->geographicalId())->surface().toGlobal(matchedHit->localPosition());
      if(preselection(position, superCluster, phiVsRSlope, 3)){
	tec3MHC++;
	hasLay2Hit = true;
	layer2Hits_.push_back(matchedHit);
      }
    }
    }
  }

  // We have 2 arrays of hits, combine them to form seeds
  if( hasLay1Hit && hasLay2Hit ){

    for (std::vector<const SiStripMatchedRecHit2D*>::const_iterator hit1 = layer1Hits_.begin() ; hit1!= layer1Hits_.end(); ++hit1) {
      for (std::vector<const SiStripMatchedRecHit2D*>::const_iterator hit2 = layer2Hits_.begin() ; hit2!= layer2Hits_.end(); ++hit2) {

	if(seedCounter < maxSeeds_){

	  if(checkHitsAndTSOS(hit1,hit2,scr,scz,pT,scEta)) {

	    seedCounter++;

	    recHits_.clear();

	    SiStripMatchedRecHit2D *hit;
	    hit=new SiStripMatchedRecHit2D(*(dynamic_cast <const SiStripMatchedRecHit2D *> (*hit1) ) );
	    recHits_.push_back(hit);
	    hit=new SiStripMatchedRecHit2D(*(dynamic_cast <const SiStripMatchedRecHit2D *> (*hit2) ) );
	    recHits_.push_back(hit);

	    PropagationDirection dir = alongMomentum;
	    reco::ElectronSeed seed(pts_,recHits_,dir) ;
	    reco::ElectronSeed::CaloClusterRef caloCluster(seedCluster) ;
	    seed.setCaloCluster(caloCluster) ;
	    result.push_back(seed);

	  }

	}

      }// end of hit 2 loop

    }// end of hit 1 loop

  }//end of seed making

  //Make seeds using TID Ring 3 if necessary

  if(hasLay1Hit && hasBackupHit && seedCounter == 0){

    for (std::vector<const SiStripMatchedRecHit2D*>::const_iterator hit1 = layer1Hits_.begin() ; hit1!= layer1Hits_.end(); ++hit1) {
      for (std::vector<const SiStripRecHit2D*>::const_iterator hit2 = backupLayer2Hits_.begin() ; hit2!= backupLayer2Hits_.end(); ++hit2) {

	if(seedCounter < maxSeeds_){

	  if(altCheckHitsAndTSOS(hit1,hit2,scr,scz,pT,scEta)) {

	    seedCounter++;

	    recHits_.clear();

	    SiStripMatchedRecHit2D *innerHit;
	    innerHit=new SiStripMatchedRecHit2D(*(dynamic_cast <const SiStripMatchedRecHit2D *> (*hit1) ) );
	    recHits_.push_back(innerHit);
	    SiStripRecHit2D *outerHit;
	    outerHit=new SiStripRecHit2D(*(dynamic_cast <const SiStripRecHit2D *> (*hit2) ) );
	    recHits_.push_back(outerHit);

	    PropagationDirection dir = alongMomentum;
	    reco::ElectronSeed seed(pts_,recHits_,dir) ;
	    reco::ElectronSeed::CaloClusterRef caloCluster(seedCluster) ;
	    seed.setCaloCluster(caloCluster) ;
	    result.push_back(seed);

	  }

	}

      }// end of hit 2 loop

    }// end of hit 1 loop

  }// end of backup seed making

} // end of findSeedsFromCluster



bool SiStripElectronSeedGenerator::checkHitsAndTSOS(std::vector<const SiStripMatchedRecHit2D*>::const_iterator hit1,
						    std::vector<const SiStripMatchedRecHit2D*>::const_iterator hit2,
						    double rc,double zc,double pT,double scEta) {

  bool seedCutsSatisfied = false;

  using namespace std;

  GlobalPoint hit1Pos = trackerGeometryHandle->idToDet((*hit1)->geographicalId())->surface().toGlobal((*hit1)->localPosition());
  double r1 = sqrt(hit1Pos.x()*hit1Pos.x() + hit1Pos.y()*hit1Pos.y());
  double phi1 = hit1Pos.phi();
  double z1=hit1Pos.z();

  GlobalPoint hit2Pos = trackerGeometryHandle->idToDet((*hit2)->geographicalId())->surface().toGlobal((*hit2)->localPosition());
  double r2 = sqrt(hit2Pos.x()*hit2Pos.x() + hit2Pos.y()*hit2Pos.y());
  double phi2 = hit2Pos.phi();
  double z2 = hit2Pos.z();

  if(r2 > r1 && (std::abs(z2) > std::abs(z1) || std::abs(scEta) < 0.25)) {

    //Consider the circle made of IP and Hit 1; Calculate it's radius using pT

    double curv = pT*100*.877;

    //Predict phi of hit 2
    double a = (r2-r1)/(2*curv);
    double b = phiDiff(phi2,phi1);
    //UB added '=0' to avoid compiler warning
    double phiMissHit2=0;
    if(std::abs(b - a)<std::abs(b + a)) phiMissHit2 = b - a;
    if(std::abs(b - a)>std::abs(b + a)) phiMissHit2 = b + a;

    double zMissHit2 = z2 - (r2*(zc-z1)-r1*zc+rc*z1)/(rc-r1);

    double rPredHit2 = r1 + (rc-r1)/(zc-z1)*(z2-z1);
    double rMissHit2 = r2 - rPredHit2;

    int subdetector = whichSubdetector(hit2);

    bool zDiff = true;
    double zVar1 = std::abs(z1);
    double zVar2 = std::abs(z2 - z1);
    if(zVar1 > 75 && zVar1 < 95 && (zVar2 > 18 || zVar2 < 5)) zDiff = false;
    if(zVar1 > 100 && zVar1 < 110 && (zVar2 > 35 || zVar2 < 5)) zDiff = false;
    if(zVar1 > 125 && zVar1 < 150 && (zVar2 > 18 || zVar2 < 5)) zDiff = false;

    if(subdetector == 1){
      int tibExtraCut = 0;
      if(r1 > 23 && r1 < 28 && r2 > 31 && r2 < 37) tibExtraCut = 1;
      if(std::abs(phiMissHit2) < tibPhiMissHit2Cut_ && std::abs(zMissHit2) < tibZMissHit2Cut_ && tibExtraCut == 1) seedCutsSatisfied = true;
    }else if(subdetector == 2){
      int tidExtraCut = 0;
      if(r1 > 23 && r1 < 34 && r2 > 26 && r2 < 42) tidExtraCut = 1;
      if(std::abs(phiMissHit2) < tidPhiMissHit2Cut_ && std::abs(rMissHit2) < tidRMissHit2Cut_ && tidExtraCut == 1 && zDiff) seedCutsSatisfied = true;
    }else if(subdetector == 3){
      int tecExtraCut = 0;
      if(r1 > 23 && r1 < 32 && r2 > 26 && r2 < 42) tecExtraCut = 1;
      if(std::abs(phiMissHit2) < tecPhiMissHit2Cut_ && std::abs(rMissHit2) < tecRMissHit2Cut_ && tecExtraCut == 1 && zDiff) seedCutsSatisfied = true;
    }

  }

  if(!seedCutsSatisfied) return false;

  // seed checks borrowed from pixel-based algoritm



  typedef TrajectoryStateOnSurface TSOS;

  double vertexZ = z1 - (r1 * (zc - z1) ) / (rc - r1);
  GlobalPoint eleVertex(0.,0.,vertexZ);

   // FIXME optimize: move outside loop
    edm::ESHandle<MagneticField> bfield;
    theSetup->get<IdealMagneticFieldRecord>().get(bfield);
    float nomField = bfield->nominalValue();

  // make a spiral
  FastHelix helix(hit2Pos,hit1Pos,eleVertex,nomField,&*bfield);
  if (!helix.isValid()) return false;

  FreeTrajectoryState fts(helix.stateAtVertex());
  TSOS propagatedState = thePropagator->propagate(fts,(*hit1)->det()->surface());

  if (!propagatedState.isValid()) return false;

  TSOS updatedState = theUpdator->update(propagatedState, **hit1);
  TSOS propagatedState_out = thePropagator->propagate(fts,(*hit2)->det()->surface()) ;

  if (!propagatedState_out.isValid()) return false;

  // the seed has now passed all the cuts

  TSOS updatedState_out = theUpdator->update(propagatedState_out, **hit2);

  pts_ =  trajectoryStateTransform::persistentState(updatedState_out, (*hit2)->geographicalId().rawId());

  return true;
}

bool SiStripElectronSeedGenerator::altCheckHitsAndTSOS(std::vector<const SiStripMatchedRecHit2D*>::const_iterator hit1,
						       std::vector<const SiStripRecHit2D*>::const_iterator hit2,
						       double rc,double zc,double pT,double scEta) {

  bool seedCutSatisfied = false;

  using namespace std;

  GlobalPoint hit1Pos = trackerGeometryHandle->idToDet((*hit1)->geographicalId())->surface().toGlobal((*hit1)->localPosition());
  double r1 = sqrt(hit1Pos.x()*hit1Pos.x() + hit1Pos.y()*hit1Pos.y());
  double phi1 = hit1Pos.phi();
  double z1=hit1Pos.z();

  GlobalPoint hit2Pos = trackerGeometryHandle->idToDet((*hit2)->geographicalId())->surface().toGlobal((*hit2)->localPosition());
  double r2 = sqrt(hit2Pos.x()*hit2Pos.x() + hit2Pos.y()*hit2Pos.y());
  double phi2 = hit2Pos.phi();
  double z2 = hit2Pos.z();

  if(r2 > r1 && std::abs(z2) > std::abs(z1)) {

    //Consider the circle made of IP and Hit 1; Calculate it's radius using pT

    double curv = pT*100*.877;

    //Predict phi of hit 2
    double a = (r2-r1)/(2*curv);
    double b = phiDiff(phi2,phi1);
    double phiMissHit2 = 0;
    if(std::abs(b - a)<std::abs(b + a)) phiMissHit2 = b - a;
    if(std::abs(b - a)>std::abs(b + a)) phiMissHit2 = b + a;

    if(std::abs(phiMissHit2) < monoPhiMissHit2Cut_) seedCutSatisfied = true;

  }

  if(!seedCutSatisfied) return false;

  // seed checks borrowed from pixel-based algoritm

 


  typedef TrajectoryStateOnSurface TSOS;

  double vertexZ = z1 - (r1 * (zc - z1) ) / (rc - r1);
  GlobalPoint eleVertex(0.,0.,vertexZ);

   // FIXME optimize: move outside loop
    edm::ESHandle<MagneticField> bfield;
    theSetup->get<IdealMagneticFieldRecord>().get(bfield);
    float nomField = bfield->nominalValue();

  // make a spiral
  FastHelix helix(hit2Pos,hit1Pos,eleVertex,nomField,&*bfield);
  if (!helix.isValid()) return false;

  FreeTrajectoryState fts(helix.stateAtVertex());
  TSOS propagatedState = thePropagator->propagate(fts,(*hit1)->det()->surface());

  if (!propagatedState.isValid()) return false;

  TSOS updatedState = theUpdator->update(propagatedState, **hit1);
  TSOS propagatedState_out = thePropagator->propagate(fts,(*hit2)->det()->surface()) ;

  if (!propagatedState_out.isValid()) return false;

  // the seed has now passed all the cuts

  TSOS updatedState_out = theUpdator->update(propagatedState_out, **hit2);

  pts_ =  trajectoryStateTransform::persistentState(updatedState_out, (*hit2)->geographicalId().rawId());

  return true;
}


bool SiStripElectronSeedGenerator::preselection(GlobalPoint position,GlobalPoint superCluster,double phiVsRSlope,int hitLayer){
  double r = position.perp();
  double phi = position.phi();
  double z = position.z();
  double scr = superCluster.perp();
  double scphi = superCluster.phi();
  double scz = superCluster.z();
  double psi = phiDiff(phi,scphi);
  double deltaPsi = psi - (scr-r)*phiVsRSlope;
  double antiDeltaPsi = psi - (r-scr)*phiVsRSlope;
  double dP;
  if (std::abs(deltaPsi)<std::abs(antiDeltaPsi)){
    dP = deltaPsi;
  }else{
    dP = antiDeltaPsi;
  }
  double originZ = (scr*z - r*scz)/(scr-r);

  bool result = false;

  if(hitLayer == 1){
    if(std::abs(originZ) < tibOriginZCut_ && std::abs(dP) < tibDeltaPsiCut_) result = true;
  }else if(hitLayer == 2){
    if(std::abs(originZ) < tidOriginZCut_ && std::abs(dP) < tidDeltaPsiCut_) result = true;
  }else if(hitLayer == 3){
    if(std::abs(originZ) < tecOriginZCut_ && std::abs(dP) < tecDeltaPsiCut_) result = true;
  }else if(hitLayer == 4){
    if(std::abs(originZ) < monoOriginZCut_ && std::abs(dP) < monoDeltaPsiCut_) result = true;
  }

  return result;
}

// Helper algorithms

int SiStripElectronSeedGenerator::whichSubdetector(std::vector<const SiStripMatchedRecHit2D*>::const_iterator hit){
  int result = 0;
  if(((*hit)->geographicalId()).subdetId() == StripSubdetector::TIB){
    result = 1;
  }else if(((*hit)->geographicalId()).subdetId() == StripSubdetector::TID){
    result = 2;
  }else if(((*hit)->geographicalId()).subdetId() == StripSubdetector::TEC){
    result = 3;
  }
  return result;
}

const SiStripMatchedRecHit2D* SiStripElectronSeedGenerator::matchedHitConverter(ConstRecHitPointer crhp){
  const TrackingRecHit* trh = crhp->hit();
  const SiStripMatchedRecHit2D* matchedHit = dynamic_cast<const SiStripMatchedRecHit2D*>(trh);
  return matchedHit;
}

const SiStripRecHit2D* SiStripElectronSeedGenerator::backupHitConverter(ConstRecHitPointer crhp){
  const TrackingRecHit* trh = crhp->hit();
  const SiStripRecHit2D* backupHit = dynamic_cast<const SiStripRecHit2D*>(trh);
  return backupHit;
}

std::vector<bool> SiStripElectronSeedGenerator::useDetLayer(double scEta){
  std::vector<bool> useDetLayer;
  double variable = std::abs(scEta);
  if(variable > 0 && variable < 1.8){
    useDetLayer.push_back(true);
  }else{
    useDetLayer.push_back(false);
  }
  if(variable > 0 && variable < 1.5){
    useDetLayer.push_back(true);
  }else{
    useDetLayer.push_back(false);
  }
  if(variable > 1 && variable < 2.1){
    useDetLayer.push_back(true);
  }else{
    useDetLayer.push_back(false);
  }
  if(variable > 1 && variable < 2.2){
    useDetLayer.push_back(true);
  }else{
    useDetLayer.push_back(false);
  }
  if(variable > 1 && variable < 2.3){
    useDetLayer.push_back(true);
  }else{
    useDetLayer.push_back(false);
  }
  if(variable > 1.8 && variable < 2.5){
    useDetLayer.push_back(true);
  }else{
    useDetLayer.push_back(false);
  }
  if(variable > 1.8 && variable < 2.5){
    useDetLayer.push_back(true);
  }else{
    useDetLayer.push_back(false);
  }
  if(variable > 1.8 && variable < 2.5){
    useDetLayer.push_back(true);
  }else{
    useDetLayer.push_back(false);
  }
  if(variable > 1.2 && variable < 1.6){
    useDetLayer.push_back(true);
  }else{
    useDetLayer.push_back(false);
  }
  return useDetLayer;
}



