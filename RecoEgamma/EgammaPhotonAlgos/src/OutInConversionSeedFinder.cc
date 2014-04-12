#include "RecoEgamma/EgammaPhotonAlgos/interface/OutInConversionSeedFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionBarrelEstimator.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionForwardEstimator.h"
#include "DataFormats/Math/interface/deltaPhi.h"
//
#include "FWCore/MessageLogger/interface/MessageLogger.h"
// Field
#include "MagneticField/Engine/interface/MagneticField.h"
//
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
// Geometry
//
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
//

//
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Geometry/Point3D.h"
#include "CLHEP/Geometry/Vector3D.h" 
 #include "CLHEP/Geometry/Transform3D.h"  
#include <cfloat>

namespace {
  inline double ptFast( const double energy, 
			const math::XYZPoint& position,
			const math::XYZPoint& origin ) {
    const auto v = position - origin;
    return energy*std::sqrt(v.perp2()/v.mag2());
  }
}


OutInConversionSeedFinder::OutInConversionSeedFinder( const edm::ParameterSet& conf ): ConversionSeedFinder( conf ), conf_(conf)  
{

  LogDebug("OutInConversionSeedFinder") << "OutInConversionSeedFinder CTOR " << "\n";      

  maxNumberOfOutInSeedsPerBC_ =  conf_.getParameter<int>("maxNumOfSeedsOutIn");
  bcEtcut_ =  conf_.getParameter<double>("bcEtCut");
  bcEcut_ =  conf_.getParameter<double>("bcECut");
  useEtCut_ =  conf_.getParameter<bool>("useEtCut");
   //the2ndHitdphi_ = 0.01; 
  the2ndHitdphi_ = 0.03; 
  the2ndHitdzConst_ = 5.;
  the2ndHitdznSigma_ = 2.;

   
    
}
 



OutInConversionSeedFinder::~OutInConversionSeedFinder() {
  LogDebug("OutInConversionSeedFinder") << "OutInConversionSeedFinder DTOR " << "\n";
 
}



void OutInConversionSeedFinder::makeSeeds( const edm::Handle<edm::View<reco::CaloCluster> > &  allBC )  const  {

  theSeeds_.clear();
  
  //  std::cout  << "  OutInConversionSeedFinder::makeSeeds() " << "\n";

  // debug  
  //  std::cout << "  OutInConversionSeedFinder::makeSeeds() SC position " << theSCPosition_.x() << " " << theSCPosition_.y() << " " << theSCPosition_.z() << "\n";
  // std::cout << " SC eta  " <<  theSCPosition_.eta() <<  " SC phi  " <<  theSCPosition_.phi() << "\n";  
  // std::cout << "  OutInConversionSeedFinder::makeSeeds() SC energy " << theSCenergy_  << "\n";  
  //

  findLayers();

  
  //  std::cout  << " Check Calo cluster collection size " << allBC->size() << "\n";
  
  float  theSCPhi=theSCPosition_.phi();
  float  theSCEta=theSCPosition_.eta();

  

  //  Loop over the Calo Clusters  in the event looking for seeds 
  reco::CaloClusterCollection::const_iterator bcItr;
  LogDebug("OutInConversionSeedFinder") << "  OutInConversionSeedFinder::makeSeeds() All BC in the event " << "\n";
  for (unsigned i = 0; i < allBC->size(); ++i ) {
  
    //for(bcItr = allBC.begin(); bcItr != allBC.end(); bcItr++) {
    nSeedsPerBC_=0;

    const reco::CaloCluster& theBC = allBC->at(i);
    const math::XYZPoint& rawBCpos = theBC.position();

    theBCPosition_ = GlobalPoint( rawBCpos.x(), rawBCpos.y(), rawBCpos.z() ) ;
    float theBcEta=  theBCPosition_.eta();
    float theBcPhi=  theBCPosition_.phi();
    //    float  dPhi= theBcPhi-theSCPhi;
    theBCEnergy_=theBC.energy();        

    float EtOrECut = bcEcut_;
    if ( useEtCut_ ) {
      theBCEnergy_= ptFast(theBCEnergy_,rawBCpos,math::XYZPoint(0,0,0));
      EtOrECut = bcEtcut_;
    }   

    if ( theBCEnergy_ < EtOrECut ) continue;
    // std::cout << "  OutInConversionSeedFinder::makeSeeds() BC eta  " << theBcEta << " phi " <<  theBcPhi << " BC transverse energy " << theBCEnergy_ << " dPhi " << fabs(theBcPhi-theSCPhi) << " dEta " <<  fabs(theBcEta-theSCEta) << "\n";

    LogDebug("OutInConversionSeedFinder") << "  OutInConversionSeedFinder::makeSeeds() Passing the >=1.5 GeV cut  BC eta  " << theBcEta << " phi " <<  theBcPhi << " BC energy " << theBCEnergy_ << "\n";

    if (  fabs(theBcEta-theSCEta) < 0.015  && reco::deltaPhi(theBcPhi,theSCPhi) < 0.3 ) { 
      LogDebug("OutInConversionSeedFinder") << "  OutInConversionSeedFinder::makeSeeds() in et and phi range passed to the analysis " << "\n";
      fillClusterSeeds( allBC->ptrAt(i)  );
    }
    

  }


  //  std::cout << "Built vector of seeds of size  " << theSeeds_.size() <<  "\n" ;
  
  ///// This part is only for local debugging: will be trhown away when no longer needed
  /*
  int nSeed=0;
  for ( std::vector<TrajectorySeed>::const_iterator iSeed= theSeeds_.begin(); iSeed != theSeeds_.end(); ++iSeed) {
    nSeed++;
    PTrajectoryStateOnDet  ptsod=iSeed->startingState();
    LogDebug("OutInConversionSeedFinder") << nSeed << ")  Direction " << iSeed->direction() << " Num of hits " << iSeed->nHits() <<  " starting state position " << ptsod.parameters().position() << " R " << ptsod.parameters().position().perp() << " phi " << ptsod.parameters().position().phi() << " eta " << ptsod.parameters().position().eta() << "\n" ;
    
    
    DetId tmpId = DetId( iSeed->startingState().detId());
    const GeomDet* tmpDet  = this->getMeasurementTracker()->geomTracker()->idToDet( tmpId );
    GlobalVector gv = tmpDet->surface().toGlobal( iSeed->startingState().parameters().momentum() );
    
    LogDebug("OutInConversionSeedFinder") << "seed perp,phi,eta : " 
					  << gv.perp() << " , " 
					  << gv.phi() << " , " 
					  << gv.eta() << "\n" ; ;
    



    TrajectorySeed::range hitRange = iSeed->recHits();
    for (TrajectorySeed::const_iterator ihit = hitRange.first; ihit != hitRange.second; ihit++) {
   
      if ( ihit->isValid() ) {

	LogDebug("OutInConversionSeedFinder") << " Valid hit global position " << this->getMeasurementTracker()->geomTracker()->idToDet((ihit)->geographicalId())->surface().toGlobal((ihit)->localPosition()) << " R " << this->getMeasurementTracker()->geomTracker()->idToDet((ihit)->geographicalId())->surface().toGlobal((ihit)->localPosition()).perp() << " phi " << this->getMeasurementTracker()->geomTracker()->idToDet((ihit)->geographicalId())->surface().toGlobal((ihit)->localPosition()).phi() << " eta " << this->getMeasurementTracker()->geomTracker()->idToDet((ihit)->geographicalId())->surface().toGlobal((ihit)->localPosition()).eta() <<    "\n" ;

      }
    }
  } 
  
  */

  
  
}



void OutInConversionSeedFinder::makeSeeds( const reco::CaloClusterPtr&  aBC )  const  {

  theSeeds_.clear();

  findLayers();

  float  theSCPhi=theSCPosition_.phi();
  float  theSCEta=theSCPosition_.eta();

  nSeedsPerBC_=0;

  // theBCEnergy_=aBC->energy();
  theBCEnergy_= ptFast(aBC->energy(),aBC->position(),math::XYZPoint(0,0,0));
  theBCPosition_ = GlobalPoint(aBC->position().x(), aBC->position().y(), aBC->position().z() ) ;
  float theBcEta=  theBCPosition_.eta();
  float theBcPhi=  theBCPosition_.phi();
  //  float  dPhi= theBcPhi-theSCPhi;

  if ( theBCEnergy_ <  bcEtcut_ ) return;

  if (  fabs(theBcEta-theSCEta) < 0.015  && fabs(theBcPhi-theSCPhi) < 0.25 ) {
    fillClusterSeeds( aBC);
  }

}


 


void OutInConversionSeedFinder::fillClusterSeeds(const reco::CaloClusterPtr& bc) const {

  
  theFirstMeasurements_.clear();
  FreeTrajectoryState fts;  

  /// negative charge state
  if ( makeTrackState(-1).second ) {
    fts = makeTrackState(-1).first;
    startSeed(fts);
  }
  /// positive charge state

  if ( makeTrackState(1).second ) {
    fts = makeTrackState(1).first;
    startSeed(fts);
  }
  theFirstMeasurements_.clear();
}



std::pair<FreeTrajectoryState,bool>  OutInConversionSeedFinder::makeTrackState(int  charge) const {

  std::pair<FreeTrajectoryState,bool> result;
  result.second=false;
 

  //std::cout << "  OutInConversionSeedFinder:makeTrackState " << "\n";


  //  Old GlobalPoint gpOrigine(theBCPosition_.x()*0.3, theBCPosition_.y()*0.3, theBCPosition_.z()*0.3) ;
 //  GlobalPoint gpOrigine(0.,0.,0.);

  GlobalPoint  gpOrigine(theBeamSpot_.position().x(),theBeamSpot_.position().y(),theBeamSpot_.position().z()); 
  GlobalVector gvBcRadius = theBCPosition_ - gpOrigine ;
  HepGeom::Point3D<double>  radiusBc(gvBcRadius.x(),gvBcRadius.y(),gvBcRadius.z()) ;
  HepGeom::Point3D<double>  momentumWithoutCurvature = radiusBc.unit() * theBCEnergy_ ;

  // compute momentum direction at calo
  double curvature = theMF_->inTesla(theBCPosition_).z() * c_light * 1.e-3 / momentumWithoutCurvature.perp() ;
  curvature /= 100. ; // in cm-1 !!

  LogDebug("OutInConversionSeedFinder") << "OutInConversionSeedFinder::makeTrackState gpOrigine " << gpOrigine.x() << " " <<  gpOrigine.y() << " " <<  gpOrigine.z() << " momentumWithoutCurvature " << momentumWithoutCurvature.mag() << " curvature " << curvature << "\n";

  // define rotation angle
  float R = theBCPosition_.perp();
  float r = gpOrigine.perp();
  float rho = 1./curvature;
  // from the formula for the intersection of two circles
  // turns out to be about 2/3 of the deflection of the old formula
  float d = sqrt(r*r+rho*rho);
  float u = rho + rho/d/d*(R*R-rho*rho) - r/d/d*sqrt((R*R-r*r+2*rho*R)*(R*R-r*r+2*rho*R));
  //float u = rho + rho/d/d*(R*R-rho*rho) ;
  if ( u <=R )   result.second=true;

  double sinAlpha = 0.5*u/R;
  if ( sinAlpha>(1.-10*DBL_EPSILON) )  sinAlpha = 1.-10*DBL_EPSILON;
  else if ( sinAlpha<-(1.-10*DBL_EPSILON) )  sinAlpha = -(1.-10*DBL_EPSILON);
  
  double newdphi = charge * asin( sinAlpha) ;

  LogDebug("OutInConversionSeedFinder") << "OutInConversionSeedFinder::makeTrackState charge " << charge << " R " << R << " u/R " << u/R << " asin(0.5*u/R) " << asin(sinAlpha) << "\n";

  HepGeom::Transform3D rotation =  HepGeom::Rotate3D(newdphi, HepGeom::Vector3D<double> (0., 0. ,1.));


  HepGeom::Point3D<double>  momentumInTracker = momentumWithoutCurvature.transform(rotation) ;
  LogDebug("OutInConversionSeedFinder") << "OutInConversionSeedFinder::makeTrackState  R " << R << " r " << r << " rho " << rho  << " d " << d  << " u " << u << " newdphi " << newdphi << " momentumInTracker " <<  momentumInTracker << "\n";

  HepGeom::Point3D<double>  hepStartingPoint(gpOrigine.x(), gpOrigine.y(), gpOrigine.z()) ;

  LogDebug("OutInConversionSeedFinder") << "OutInConversionSeedFinder::makeTrackState hepStartingPoint " << hepStartingPoint << "\n";

  hepStartingPoint.transform(rotation);

  GlobalPoint startingPoint(hepStartingPoint.x(), hepStartingPoint.y(), hepStartingPoint.z());

  LogDebug("OutInConversionSeedFinder") << "OutInConversionSeedFinder::makeTrackState startingPoint " << startingPoint << " calo position " << theBCPosition_ << "\n";
  GlobalVector gvTracker(momentumInTracker.x(), momentumInTracker.y(), momentumInTracker.z());
  GlobalTrajectoryParameters gtp(startingPoint, gvTracker, charge, &(*theMF_) );
  // error matrix
  AlgebraicSymMatrix55 m = AlgebraicMatrixID();
  m(0,0) = 0.1; m(1,1) = 0.1 ; m(2,2) = 0.1 ;
  m(3,3) = 0.1 ; m(4,4) = 0.1;
  
  //  std::cout << "OutInConversionSeedFinder::makeTrackState " <<  FreeTrajectoryState(gtp, CurvilinearTrajectoryError(m) ) << "\n";
   
  result.first= FreeTrajectoryState(gtp, CurvilinearTrajectoryError(m) ) ;
  return result;

}


void OutInConversionSeedFinder::startSeed(const FreeTrajectoryState & fts) const {


  //  std::cout << "OutInConversionSeedFinder::startSeed layer list " << this->layerList().size() <<  "\n";
  //std::cout << "OutInConversionSeedFinder::startSeed  fts " << fts <<  "\n";  

  std::vector<const DetLayer*> myLayers=layerList();
  if ( myLayers.size() > 3 ) {
    
    for(unsigned int ilayer = myLayers.size()-1; ilayer >= myLayers.size()-2; --ilayer) {
      const DetLayer * layer = myLayers[ilayer];
      
      
      // allow the z of the hit to be within a straight line from a vertex
      // of +-15 cm to the cluster
      //      float dphi = 0.015;
      float dphi = 0.030;

      MeasurementEstimator * newEstimator = makeEstimator(layer, dphi);

     
      //std::cout << "OutInSeedFinder::startSeed propagationDirection  " << int(thePropagatorAlongMomentum_->propagationDirection() ) << "\n";       
      
      TSOS tsos(fts, layer->surface() );
      
      LogDebug("OutInConversionSeedFinder") << "OutInSeedFinder::startSeed  after  TSOS tsos(fts, layer->surface() ) " << "\n";
       
      LayerMeasurements theLayerMeasurements_( *this->getMeasurementTracker(), *theTrackerData_ );
      theFirstMeasurements_ = theLayerMeasurements_.measurements( *layer, tsos, *thePropagatorAlongMomentum_, *newEstimator);
      
      //std::cout << "OutInSeedFinder::startSeed  after  theFirstMeasurements_   " << theFirstMeasurements_.size() <<  "\n";
      
      if(theFirstMeasurements_.size() > 1) // always a dummy returned, too
	LogDebug("OutInConversionSeedFinder") <<  " Found " << theFirstMeasurements_.size()-1 << " 1st hits in seed" << "\n";
      
      delete newEstimator;
      
      LogDebug("OutInConversionSeedFinder") << "OutInSeedFinder::startSeed Layer " << ilayer << " theFirstMeasurements_.size " << theFirstMeasurements_.size() << "\n";
      
      for(unsigned int i = 0; i < theFirstMeasurements_.size(); ++i) {
	TrajectoryMeasurement m1 = theFirstMeasurements_[i];
	if(m1.recHit()->isValid()) {
	  
	  // update the fts to start from this point.  much better than starting from
	  // extrapolated point along the line
	  GlobalPoint hitPoint = m1.recHit()->globalPosition();
	  LogDebug("OutInConversionSeedFinder")  << " Valid hit at R  " <<   m1.recHit()->globalPosition().perp() << " Z " <<  m1.recHit()->globalPosition().z() << " eta " << m1.recHit()->globalPosition().eta() << " phi " << m1.recHit()->globalPosition().phi()  << " xyz " <<  m1.recHit()->globalPosition() << "\n";

	  
	  FreeTrajectoryState newfts = trackStateFromClusters(fts.charge(), hitPoint, alongMomentum, 0.8);
	  //std::cout << "OutInConversionSeedFinder::startSeed  newfts " << newfts << "\n";
	  LogDebug("OutInConversionSeedFinder") << "OutInConversionSeedFinder::startSeed  newfts " << newfts << "\n";
	  LogDebug("OutInConversionSeedFinder") << "OutInConversionSeedFinder::startSeed propagationDirection  after switching " << int(thePropagatorOppositeToMomentum_->propagationDirection() ) << "\n";        
	  //  std::cout << "OutInConversionSeedFinder::startSeed propagationDirection  after switching " << int(thePropagatorOppositeToMomentum_->propagationDirection() ) << "\n";        

       
	  completeSeed(m1, newfts, thePropagatorOppositeToMomentum_, ilayer-1);
	  // skip a layer, if you haven't already skipped the first layer
	  if(ilayer == myLayers.size()-1) {
	    completeSeed(m1, newfts, thePropagatorOppositeToMomentum_, ilayer-2);
	  }
	}
      }
      
    } // loop over layers
  }



  
}



MeasurementEstimator * OutInConversionSeedFinder::makeEstimator(const DetLayer * layer, float dphi) const {
 
  //std::cout  << "OutInConversionSeedFinder::makeEstimator  " << "\n";

  MeasurementEstimator * newEstimator=0;

  if (layer->location() == GeomDetEnumerators::barrel ) {
    
    const BarrelDetLayer * barrelLayer = dynamic_cast<const BarrelDetLayer*>(layer);
    LogDebug("OutInConversionSeedFinder") << "OutInConversionSeedFinder::makeEstimator Barrel  r = " << barrelLayer->specificSurface().radius() << " " << "\n";        
    float r = barrelLayer->specificSurface().radius();
    float zrange = 15.* (1.-r/theBCPosition_.perp());
    newEstimator = new ConversionBarrelEstimator(-dphi, dphi, -zrange, zrange);
  }



  if (layer->location() == GeomDetEnumerators::endcap ) {   
   
    const ForwardDetLayer * forwardLayer = dynamic_cast<const ForwardDetLayer*>(layer);
    LogDebug("OutInConversionSeedFinder") << "OutInConversionSeedFinder::makeEstimator Endcap r = " << forwardLayer->specificSurface().innerRadius() << " R " << forwardLayer->specificSurface().outerRadius()  <<  " Z " << forwardLayer->specificSurface().position().z() << "\n";  
    
    float zc = fabs(theBCPosition_.z());
    float z =  fabs(forwardLayer->surface().position().z());
    
    float rrange = 15. * theBCPosition_.perp() * (zc - z) / (zc*zc - 15.*zc);
    newEstimator = new ConversionForwardEstimator(-dphi, dphi, rrange);
  }




  return newEstimator;
}




void OutInConversionSeedFinder::completeSeed(const TrajectoryMeasurement & m1, 
					     FreeTrajectoryState & fts, 
					     const Propagator* propagator, int ilayer) const {

  //std::cout <<  "OutInConversionSeedFinder::completeSeed ilayer " << ilayer << "\n";

  MeasurementEstimator * newEstimator=0;
  const DetLayer * layer = theLayerList_[ilayer];
  

  if ( layer->location() == GeomDetEnumerators::barrel ) {
    // z error for 2nd hit is  2 sigma quadded with 5 cm
    LogDebug("OutInConversionSeedFinder") << " Barrel OutInConversionSeedFinder::completeSeed " << the2ndHitdznSigma_ << " " << the2ndHitdzConst_ << " " << the2ndHitdphi_ << "\n";
    float dz = sqrt(the2ndHitdznSigma_*the2ndHitdznSigma_*m1.recHit()->globalPositionError().czz() 
		    + the2ndHitdzConst_*the2ndHitdzConst_);
    newEstimator =
      new ConversionBarrelEstimator(-the2ndHitdphi_, the2ndHitdphi_, -dz, dz);
  }
  else {
    LogDebug("OutInConversionSeedFinder") << " EndCap OutInConversionSeedFinder::completeSeed " << the2ndHitdznSigma_ << " " << the2ndHitdzConst_ << " " << the2ndHitdphi_ << "\n";
    // z error for 2nd hit is 2sigma quadded with 5 cm
    //float m1dr = m1.recHit().globalPositionError().rerr(m1.recHit().globalPosition());
    float m1dr = sqrt(m1.recHit()->localPositionError().yy());
    float dr = sqrt(the2ndHitdznSigma_*the2ndHitdznSigma_*m1dr*m1dr 
		    + the2ndHitdzConst_*the2ndHitdznSigma_);
    // LogDebug("OutInConversionSeedFinder") << "second hit forward dr " << dr << " this hit " << m1dr << endl;
    newEstimator =
      new ConversionForwardEstimator(-the2ndHitdphi_, the2ndHitdphi_, dr);
  }

  LogDebug("OutInConversionSeedFinder") << "OutInConversionSeedFinder::completeSeed  ilayer " << ilayer <<  "\n"; 
  
  // Get the measurements consistent with the FTS and the Estimator
  TSOS tsos(fts, layer->surface() );
  LogDebug("OutInConversionSeedFinder") << "OutInConversionSeedFinder::completeSeed propagationDirection  " << int(propagator->propagationDirection() ) << "\n";               
  LogDebug("OutInConversionSeedFinder") << "OutInConversionSeedFinder::completeSeed pointer to estimator " << newEstimator << "\n";

  LayerMeasurements theLayerMeasurements_( *this->getMeasurementTracker(), *theTrackerData_ );
  std::vector<TrajectoryMeasurement> measurements = theLayerMeasurements_.measurements( *layer, tsos, *propagator, *newEstimator);
  //std::cout << "OutInConversionSeedFinder::completeSeed Found " << measurements.size() << " second hits " << "\n";
  delete newEstimator;

  for(unsigned int i = 0; i < measurements.size(); ++i) {
    if( measurements[i].recHit()->isValid()  ) {
      createSeed(m1, measurements[i]);
    }
  }



  //LogDebug("OutInConversionSeedFinder") << "COMPLETED " << theSeeds_.size() << " SEEDS " << "\n";
}



void OutInConversionSeedFinder::createSeed(const TrajectoryMeasurement & m1, 
					   const TrajectoryMeasurement & m2) const {

  //std::cout  << "OutInConversionSeedFinder::createSeed  from hit1 " << m1.recHit()->globalPosition() << " r1 " << m1.recHit()->globalPosition().perp() << " and hit2 " << m2.recHit()->globalPosition() << " r2 " << m2.recHit()->globalPosition().perp() << "\n";
  

  FreeTrajectoryState fts = createSeedFTS(m1, m2);


  //std::cout << "OutInConversionSeedFinder::createSeed First point errors " <<m1.recHit()->parametersError() << "\n";
  // std::cout << "original cluster FTS " << fts <<"\n";
  LogDebug("OutInConversionSeedFinder") << "OutInConversionSeedFinder::createSeed First point errors " <<m1.recHit()->parametersError() << "\n";
  LogDebug("OutInConversionSeedFinder") << "original cluster FTS " << fts <<"\n";



  //std::cout  << "OutInConversionSeedFinder::createSeed propagation dir " << int( thePropagatorOppositeToMomentum_->propagationDirection() ) << "\n"; 
  TrajectoryStateOnSurface state1 = thePropagatorOppositeToMomentum_->propagate(fts,  m1.recHit()->det()->surface());

  // LogDebug("OutInConversionSeedFinder") << "hit surface " << h1.det().surface().position() << endl;
  // LogDebug("OutInConversionSeedFinder") << "prop to " << typeid(h1.det().surface()).name() << endl;
  // LogDebug("OutInConversionSeedFinder") << "prop to first hit " << state1 << endl; 
  // LogDebug("OutInConversionSeedFinder") << "update to " << h1.globalPosition() << endl;

  if ( state1.isValid() ) {
    TrajectoryStateOnSurface updatedState1 = theUpdator_.update(state1,  *m1.recHit() );

    if ( updatedState1.isValid() ) {
      TrajectoryStateOnSurface state2 = thePropagatorOppositeToMomentum_->propagate(*updatedState1.freeTrajectoryState(),  m2.recHit()->det()->surface());

      if ( state2.isValid() ) {

	TrajectoryStateOnSurface updatedState2 = theUpdator_.update(state2, *m2.recHit() );
	TrajectoryMeasurement meas1(state1, updatedState1,  m1.recHit()  , m1.estimate(), m1.layer());
	TrajectoryMeasurement meas2(state2, updatedState2,  m2.recHit()  , m2.estimate(), m2.layer());
	
        edm::OwnVector<TrackingRecHit> myHits;
	myHits.push_back(meas1.recHit()->hit()->clone());
	myHits.push_back(meas2.recHit()->hit()->clone());

        if ( nSeedsPerBC_ >= maxNumberOfOutInSeedsPerBC_ ) return;
   
	
	PTrajectoryStateOnDet  ptsod= trajectoryStateTransform::persistentState(state2, meas2.recHit()->hit()->geographicalId().rawId()  );

	LogDebug("OutInConversionSeedFinder") << "OutInConversionSeedFinder::createSeed new seed  from state " << state2.globalPosition()  <<  "\n";
	LogDebug("OutInConversionSeedFinder") << "OutInConversionSeedFinder::createSeed new seed  ptsod " <<  ptsod.parameters().position() << " R " 
                                              << ptsod.parameters().position().perp() << " phi " << ptsod.parameters().position().phi() << " eta " 
                                              << ptsod.parameters().position().eta() << "\n";
	


	theSeeds_.push_back(TrajectorySeed( ptsod, myHits, oppositeToMomentum ));
        nSeedsPerBC_++;

      }
    }
  }





}





FreeTrajectoryState OutInConversionSeedFinder::createSeedFTS(const TrajectoryMeasurement & m1,
							     const TrajectoryMeasurement & m2) const {



  GlobalPoint xmeas = fixPointRadius(m1);
  GlobalPoint xvert = fixPointRadius(m2);


  float pt = theSCenergy_ * sin(theSCPosition_.theta());
  float pz = theSCenergy_ * cos(theSCPosition_.theta());



  // doesn't work at all for endcap, where r is badly measured
  //float dphidr = (p1.phi()-p2.phi())/(p1.perp()-p2.perp());
  //int charge = (dphidr > 0.) ? -1 : 1;
  int charge = m2.predictedState().charge();

  double BInTesla = theMF_->inTesla(xmeas).z();
  GlobalVector xdiff = xmeas -xvert;
 
  double phi= xdiff.phi();
  double pxOld = pt*cos(phi);
  double pyOld = pt*sin(phi);
  double RadCurv = 100*pt/(BInTesla*0.29979);
  double alpha = asin(0.5*xdiff.perp()/RadCurv);
  float ca = cos(charge*alpha);
  float sa = sin(charge*alpha);
  double pxNew =   ca*pxOld + sa*pyOld;
  double pyNew =  -sa*pxOld + ca*pyOld;
  GlobalVector pNew(pxNew, pyNew, pz);

  GlobalTrajectoryParameters gp(xmeas, pNew, charge, &(*theMF_) );

  AlgebraicSymMatrix55 m = AlgebraicMatrixID();
  m(0,0) = 0.05; m(1,1) = 0.02 ; m(2,2) = 0.007 ;
  m(3,3) = 10. ; m(4,4) = 10. ;
  //std::cout  << "OutInConversionSeedFinder::createSeedFTS " <<  FreeTrajectoryState(gp, CurvilinearTrajectoryError(m))  << "\n";
  return FreeTrajectoryState(gp, CurvilinearTrajectoryError(m));


}




GlobalPoint OutInConversionSeedFinder::fixPointRadius(const  TrajectoryMeasurement& m1) const {
  GlobalPoint p1 = m1.recHit()->globalPosition();
  GlobalPoint p2;
  if(m1.layer()->location() == GeomDetEnumerators::barrel) {
    p2 = p1;
  } else {
    float z = p1.z();
    float phi = p1.phi();
    float theta = theSCPosition_.theta();
    float r = p1.z() * tan(theta);
    p2 = GlobalPoint(r*cos(phi), r*sin(phi), z);
    // LogDebug("OutInConversionSeedFinder") << "fixing point radius " << p2 << " from " << p1 << endl;
  }
  return p2;
}





