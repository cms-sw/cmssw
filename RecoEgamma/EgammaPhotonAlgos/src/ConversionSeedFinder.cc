#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionSeedFinder.h"
// Field
// Geometry
#include "Geometry/Records/interface/IdealGeometryRecord.h"
//
#include "RecoTracker/MeasurementDet/interface/StartingLayerFinder.h"
#include "RecoTracker/MeasurementDet/interface/LayerCollector.h"
//

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

ConversionSeedFinder::ConversionSeedFinder(const edm::ParameterSet& config,edm::ConsumesCollector & iC ): 
  //  conf_(config),
  theUpdator_()
{

  measurementTrkToken_=iC.consumes<MeasurementTrackerEvent>(edm::InputTag("MeasurementTrackerEvent")); //hardcoded because the original was and no time to fix (sigh)
  beamSpotToken_=iC.consumes<reco::BeamSpot>(edm::InputTag("offlineBeamSpot"));  //hardcoded because the original was and no time to fix (sigh)

  LogDebug("ConversionSeedFinder")  << " CTOR " << "\n";

  theMeasurementTrackerName_=config.getParameter<std::string>("MeasurementTrackerName");
}



void ConversionSeedFinder::setEvent(const edm::Event& evt  )  {
 
  theTrackerGeom_= this->getMeasurementTracker()->geomTracker();

  //get the BeamSpot
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  evt.getByToken(beamSpotToken_,recoBeamSpotHandle);
  theBeamSpot_ = *recoBeamSpotHandle;

  evt.getByToken(measurementTrkToken_, theTrackerData_);

}

void ConversionSeedFinder::setEventSetup(const edm::EventSetup& es  )  {
  es.get<TrackerRecoGeometryRecord>().get( theGeomSearchTracker_);
  es.get<IdealMagneticFieldRecord>().get( theMF_ );


  edm::ESHandle<MeasurementTracker> measurementTrackerHandle;
  es.get<CkfComponentsRecord>().get(theMeasurementTrackerName_,measurementTrackerHandle);
  theMeasurementTracker_ = measurementTrackerHandle.product();
  
  edm::ESHandle<Propagator>  propagatorAlongMomHandle;
  es.get<TrackingComponentsRecord>().get("alongMomElePropagator",propagatorAlongMomHandle);
  thePropagatorAlongMomentum_ = &(*propagatorAlongMomHandle);
 

  edm::ESHandle<Propagator>  propagatorOppoToMomHandle;
  es.get<TrackingComponentsRecord>().get("oppositeToMomElePropagator",propagatorOppoToMomHandle);
  thePropagatorOppositeToMomentum_ = &(*propagatorOppoToMomHandle);




}


void ConversionSeedFinder::findLayers() const {
  

  int charge;
  //List the DetLayers crossed by a straight line from the centre of the 
  //detector to the supercluster position
  //  GlobalPoint  vertex(0.,0.,0.);
  GlobalPoint  vertex(theBeamSpot_.position().x(),theBeamSpot_.position().y(),theBeamSpot_.position().z()); 
  charge=-1;  
  FreeTrajectoryState theStraightLineFTS = trackStateFromClusters(charge, vertex, alongMomentum, 1.);
  
  findLayers( theStraightLineFTS  );
  
  
}

FreeTrajectoryState ConversionSeedFinder::trackStateFromClusters( int charge, const GlobalPoint  & theOrigin, 
								  PropagationDirection dir, float scaleFactor) const {
  
  

  double caloEnergy = theSCenergy_ * scaleFactor ;
  
  GlobalVector radiusCalo = theSCPosition_ - theOrigin ;
  
  GlobalVector momentumWithoutCurvature = radiusCalo.unit() * caloEnergy;
  
  
  GlobalTrajectoryParameters gtp;
  if(dir == alongMomentum) {
    gtp = GlobalTrajectoryParameters(theOrigin, momentumWithoutCurvature, charge, &(*theMF_) ) ;
  } else {
    gtp = GlobalTrajectoryParameters(theSCPosition_, momentumWithoutCurvature, charge, &(*theMF_) ) ;
  }



  
  // now create error matrix
  // dpos = 4mm/sqrt(E), dtheta = move vertex by 1sigma
  float dpos = 0.4/sqrt(theSCenergy_);
  dpos *= 2.;
  float dphi = dpos/theSCPosition_.perp();
  //  float dp = 0.03 * sqrt(theCaloEnergy);
  //  float dp = theCaloEnergy / sqrt(12.); // for fun
  float theta1 = theSCPosition_.theta();
  float theta2 = atan2(double(theSCPosition_.perp()), theSCPosition_.z()-5.5);
  float dtheta = theta1 - theta2;
  AlgebraicSymMatrix55  m;
  m[0][0] = 1.; m[1][1] = dpos*dpos ; m[2][2] = dpos*dpos ;
  m[3][3] = dphi*dphi ; m[4][4] = dtheta * dtheta ;

  FreeTrajectoryState fts(gtp, CurvilinearTrajectoryError(m)) ;

  return fts ;


}

void ConversionSeedFinder::findLayers(const FreeTrajectoryState & traj) const {



  theLayerList_.clear();


  StraightLinePropagator prop( &(*theMF_), alongMomentum);

  StartingLayerFinder starter(&prop, this->getMeasurementTracker() );
 
  LayerCollector collector(theNavigationSchool_, &prop, &starter, 5., 5.);

  theLayerList_ = collector.allLayers(traj);

  
  for(unsigned int i = 0; i < theLayerList_.size(); ++i) {
    printLayer(i);
  }
  

}


void ConversionSeedFinder::printLayer(int i) const {
  const DetLayer * layer = theLayerList_[i];
  if (layer->location() == GeomDetEnumerators::barrel ) {
    //    const BarrelDetLayer * barrelLayer = dynamic_cast<const BarrelDetLayer*>(layer);
    //float r = barrelLayer->specificSurface().radius();
    //    std::cout   <<  " barrel layer radius " << r << " " << barrelLayer->specificSurface().bounds().length()/2. << "\n";

  } else {
    //    const ForwardDetLayer * forwardLayer = dynamic_cast<const ForwardDetLayer*>(layer);
    // float z =  fabs(forwardLayer->surface().position().z());
    //    std::cout   << " forward layer position " << z << " " << forwardLayer->specificSurface().innerRadius() << " " << forwardLayer->specificSurface().outerRadius() << "\n";
  }
}











