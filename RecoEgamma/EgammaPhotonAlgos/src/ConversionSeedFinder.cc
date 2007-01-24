#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionSeedFinder.h"
// Field
#include "MagneticField/Engine/interface/MagneticField.h"
// Geometry
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
//
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h" 
#include "RecoTracker/TkNavigation/interface/StartingLayerFinder.h"
#include "RecoTracker/TkNavigation/interface/LayerCollector.h"
//



ConversionSeedFinder::ConversionSeedFinder( const MagneticField* field, const MeasurementTracker* theInputMeasurementTracker) :
  theMF_(field), theMeasurementTracker_(theInputMeasurementTracker ), 
  theOutwardStraightPropagator_(theMF_, dir_ = alongMomentum ),
  thePropagatorWithMaterial_(dir_ = alongMomentum, 0.000511, theMF_ ), theUpdator_()

 
{

    LogDebug("ConversionSeedFinder")  << " CTOR " << "\n";
      
}



void ConversionSeedFinder::findLayers() const {

    LogDebug("ConversionSeedFinder")  << "::findLayers() " << "\n"; 
  int charge;
  //List the DetLayers crossed by a straight line from the centre of the 
  //detector to the supercluster position
  GlobalPoint  vertex(0.,0.,0.);
  charge=-1;  
  FreeTrajectoryState theStraightLineFTS = trackStateFromClusters(charge, vertex, alongMomentum, 1.);
  
  findLayers( theStraightLineFTS  );

  
}
						  
FreeTrajectoryState ConversionSeedFinder::trackStateFromClusters( int charge, const GlobalPoint  & theOrigin, 
								       PropagationDirection dir, float scaleFactor) const {


    LogDebug("ConversionSeedFinder")  << "::trackStateFromClusters " << "\n"; 
  double caloEnergy = theSC_->energy() * scaleFactor ;

  GlobalVector radiusCalo = theSCPosition_ - theOrigin ;

  GlobalVector momentumWithoutCurvature = radiusCalo.unit() * caloEnergy;


  GlobalTrajectoryParameters gtp;
  if(dir == alongMomentum) {
    gtp = GlobalTrajectoryParameters(theOrigin, momentumWithoutCurvature, charge, theMF_) ;
  } else {
    gtp = GlobalTrajectoryParameters(theSCPosition_, momentumWithoutCurvature, charge, theMF_) ;
  }



  
  // now create error matrix
  // dpos = 4mm/sqrt(E), dtheta = move vertex by 1sigma
  float dpos = 0.4/sqrt(theSC_->energy());
  dpos *= 2.;
  float dphi = dpos/theSCPosition_.perp();
//  float dp = 0.03 * sqrt(theCaloEnergy);
//  float dp = theCaloEnergy / sqrt(12.); // for fun
  float theta1 = theSCPosition_.theta();
  float theta2 = atan2(double(theSCPosition_.perp()), theSCPosition_.z()-5.5);
  float dtheta = theta1 - theta2;
  AlgebraicSymMatrix  m(5,1) ;
  m[0][0] = 1.; m[1][1] = dpos*dpos ; m[2][2] = dpos*dpos ;
  m[3][3] = dphi*dphi ; m[4][4] = dtheta * dtheta ;
//  m(1,1) = 100.; m(2,2) = 100. ; m(3,3) = 100. ;
//  m(4,4) = 100. ; m(5,5) = 100. ;


  FreeTrajectoryState fts(gtp, CurvilinearTrajectoryError(m)) ;

  return fts ;


}

void ConversionSeedFinder::findLayers(const FreeTrajectoryState & traj) const {

  theLayerList_.clear();

  StartingLayerFinder starter(&theOutwardStraightPropagator_, this->getMeasurementTracker() );
 
  LayerCollector collector(&theOutwardStraightPropagator_, &starter, 5., 5.);

  theLayerList_ = collector.allLayers(traj);
  
  for(unsigned int i = 0; i < theLayerList_.size(); ++i) {
    printLayer(i);
  }
  

}


void ConversionSeedFinder::printLayer(int i) const {
  const DetLayer * layer = theLayerList_[i];
  if (layer->location() == GeomDetEnumerators::barrel ) {
    const BarrelDetLayer * barrelLayer = dynamic_cast<const BarrelDetLayer*>(layer);
    float r = barrelLayer->specificSurface().radius();
      LogDebug("ConversionSeedFinder")  <<  " barrel layer radius " << r << " " << barrelLayer->specificSurface().bounds().length()/2. << "\n";

  } else {
    const ForwardDetLayer * forwardLayer = dynamic_cast<const ForwardDetLayer*>(layer);
    float z =  fabs(forwardLayer->surface().position().z());
      LogDebug("ConversionSeedFinder")  << " forward layer position " << z << " " << forwardLayer->specificSurface().innerRadius() << " " << forwardLayer->specificSurface().outerRadius() << "\n";
  }
}











