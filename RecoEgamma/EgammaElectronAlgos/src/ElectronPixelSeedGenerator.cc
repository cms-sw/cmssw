// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      ElectronPixelSeedGenerator.
// 
/**\class ElectronPixelSeedGenerator EgammaElectronAlgos/ElectronPixelSeedGenerator

 Description: Top algorithm producing ElectronPixelSeeds, ported from ORCA

 Implementation:
     future redesign...
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id: ElectronPixelSeedGenerator.cc,v 1.35 2008/02/13 13:27:56 uberthon Exp $
//
//
#include "RecoEgamma/EgammaElectronAlgos/interface/PixelHitMatcher.h" 
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronPixelSeedGenerator.h" 

#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h" 
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h" 
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "RecoTracker/TkNavigation/interface/SimpleNavigationSchool.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/EcalCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include <vector>
#include <utility>
ElectronPixelSeedGenerator::ElectronPixelSeedGenerator(const edm::ParameterSet &pset)
  :   dynamicphiroad_(pset.getParameter<bool>("dynamicPhiRoad")),
      SCEtCut_(pset.getParameter<double>("SCEtCut")),
      lowPtThreshold_(pset.getParameter<double>("LowPtThreshold")),
      highPtThreshold_(pset.getParameter<double>("HighPtThreshold")),
      sizeWindowENeg_(pset.getParameter<double>("SizeWindowENeg")),
      phimin2_(pset.getParameter<double>("PhiMin2")),      
      phimax2_(pset.getParameter<double>("PhiMax2")),
      deltaPhi1Low_(pset.getParameter<double>("DeltaPhi1Low")),
      deltaPhi2Low_(pset.getParameter<double>("DeltaPhi2Low")),
      deltaPhi1High_(pset.getParameter<double>("DeltaPhi1High")),
      deltaPhi2High_(pset.getParameter<double>("DeltaPhi2High")),
      myMatchEle(0), myMatchPos(0),
      theUpdator(0), thePropagator(0), theMeasurementTracker(0), 
      theNavigationSchool(0), theSetup(0), pts_(0)
{      // Instantiate the pixel hit matchers
  //       LogDebug("") << "ElectronPixelSeedGenerator, phi limits: " << ephimin1 << ", " << ephimax1 << ", "
  // 		   << pphimin1 << ", " << pphimax1;
  myMatchEle = new PixelHitMatcher( pset.getParameter<double>("ePhiMin1"), 
				    pset.getParameter<double>("ePhiMax1"),
				    pset.getParameter<double>("PhiMin2"),
				    pset.getParameter<double>("PhiMax2"),
				    pset.getParameter<double>("z2MinB"),
				    pset.getParameter<double>("z2MaxB"),
				    pset.getParameter<double>("z2MinF"),
				    pset.getParameter<double>("z2MaxF"),
				    pset.getParameter<double>("rMin"),
				    pset.getParameter<double>("rMax"));

  myMatchPos = new PixelHitMatcher( pset.getParameter<double>("pPhiMin1"),
				    pset.getParameter<double>("pPhiMax1"),
				    pset.getParameter<double>("PhiMin2"),
				    pset.getParameter<double>("PhiMax2"),
				    pset.getParameter<double>("z2MinB"),
				    pset.getParameter<double>("z2MaxB"),
				    pset.getParameter<double>("z2MinF"),
				    pset.getParameter<double>("z2MaxF"),
				    pset.getParameter<double>("rMin"),
				    pset.getParameter<double>("rMax"));




}

ElectronPixelSeedGenerator::~ElectronPixelSeedGenerator() {

  delete myMatchEle;
  delete myMatchPos;
  delete thePropagator;
  delete theUpdator;

}


void ElectronPixelSeedGenerator::setupES(const edm::EventSetup& setup) {

  theSetup= &setup;

  setup.get<IdealMagneticFieldRecord>().get(theMagField);
  setup.get<TrackerRecoGeometryRecord>().get( theGeomSearchTracker );

  edm::ESHandle<NavigationSchool> nav;
  setup.get<NavigationSchoolRecord>().get("SimpleNavigationSchool", nav);
  theNavigationSchool = nav.product();
  NavigationSetter setter(*theNavigationSchool);

  edm::ESHandle<MeasurementTracker>    measurementTrackerHandle;
  setup.get<CkfComponentsRecord>().get(measurementTrackerHandle);
  theMeasurementTracker = measurementTrackerHandle.product();

 
  if (theUpdator) delete theUpdator;
  theUpdator = new KFUpdator();
  if (thePropagator) delete thePropagator;
  thePropagator = new PropagatorWithMaterial(alongMomentum,.1057,&(*theMagField)); 
  
  myMatchEle->setES(&(*theMagField),theMeasurementTracker);
  myMatchPos->setES(&(*theMagField),theMeasurementTracker);
}

void  ElectronPixelSeedGenerator::run(edm::Event& e, const edm::EventSetup& setup, const edm::Handle<reco::SuperClusterCollection> &clusters, reco::ElectronPixelSeedCollection & out){

  //Getting the beamspot from the Event:
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  e.getByType(recoBeamSpotHandle);

  // gets its position
  BSPosition_ = recoBeamSpotHandle->position();
  double sigmaZ=recoBeamSpotHandle->sigmaZ();
  double sigmaZ0Error=recoBeamSpotHandle->sigmaZ0Error();
  double sq=sqrt(sigmaZ*sigmaZ+sigmaZ0Error*sigmaZ0Error);
  zmin1_=BSPosition_.z()-3*sq;
  zmax1_=BSPosition_.z()+3*sq;

  theSetup= &setup; 
  //  theMeasurementTracker->updatePixels(e);
  theMeasurementTracker->update(e);  //FIXME???
  
  for  (unsigned int i=0;i<clusters->size();++i) {
    edm::Ref<reco::SuperClusterCollection> theClusB(clusters,i);
    // Find the seeds
    recHits_.clear();

    LogDebug ("run") << "new cluster, calling seedsFromThisCluster";
    if (theClusB->energy()/cosh(theClusB->eta())>SCEtCut_)     seedsFromThisCluster(theClusB,out) ;
  }
  
  LogDebug ("run") << ": For event "<<e.id();
  LogDebug ("run") <<"Nr of superclusters: "<<clusters->size()
   <<", no. of ElectronPixelSeeds found  = " << out.size();
}

void ElectronPixelSeedGenerator::seedsFromThisCluster( edm::Ref<reco::SuperClusterCollection> seedCluster, reco::ElectronPixelSeedCollection& result)
{
  float Energy_factorcorrected = 0 ;
  float numSubClusters = seedCluster->clustersSize();

  if (numSubClusters > 1)
    {
      if(fabs(seedCluster->eta()) < 1.479) Energy_factorcorrected = seedCluster->energy() / fEtaBarrelBad(seedCluster->eta());
      else if(fabs(seedCluster->eta()) > 1.479) Energy_factorcorrected = seedCluster->energy() / fEtaEndcapBad(seedCluster->eta());
    }
  else if(numSubClusters == 1) 
    {
      if(fabs(seedCluster->eta()) < 1.479) Energy_factorcorrected = seedCluster->energy() / fEtaBarrelGood(seedCluster->eta());
      else if(fabs(seedCluster->eta()) > 1.479) Energy_factorcorrected = seedCluster->energy() / fEtaEndcapGood(seedCluster->eta());
    }
  
  //float clusterEnergy =  Energy_factorcorrected ;
  float clusterEnergy = seedCluster->energy();
  GlobalPoint clusterPos(seedCluster->position().x(),
			 seedCluster->position().y(), 
			 seedCluster->position().z());

  const GlobalPoint vertexPos(BSPosition_.x(),BSPosition_.y(),BSPosition_.z());
  LogDebug("") << "[ElectronPixelSeedGenerator::seedsFromThisCluster] new supercluster with energy: " << clusterEnergy;
  LogDebug("") << "[ElectronPixelSeedGenerator::seedsFromThisCluster] and position: " << clusterPos;

  myMatchEle->set1stLayerZRange(zmin1_,zmax1_);
  myMatchPos->set1stLayerZRange(zmin1_,zmax1_);
  
  // to transmit: phimin2, phimax2, Et thresholds,  windowsize
  // + ephimin1,ephimax1) pphimin1,pphimax1);
  //  phimin2, phimax2, sizeWindowENeg lowPtThreshold, highPtThreshold
  // ephimin1,ephimax1,pphimin1,pphimax1
  // Here we change the deltaPhi window of the first pixel layer in function of the seed pT
  if (dynamicphiroad_)
    {
      float clusterEnergyT = clusterEnergy*sin(seedCluster->position().theta()) ;

      float deltaPhi1 = 2.9077/clusterEnergyT - 0.003; //FIXME: constants?
      float deltaPhi2 = 0.0729/clusterEnergyT + 0.0019;

      if (clusterEnergyT < lowPtThreshold_) {
	//	 deltaPhi1=.32;
	//	 deltaPhi2=.01;
	deltaPhi1= deltaPhi1Low_;
	deltaPhi2= deltaPhi2Low_;
      }
      if (clusterEnergyT > highPtThreshold_) {
	deltaPhi1= deltaPhi1High_;
	deltaPhi2= deltaPhi2High_;
      }
      float ephimin1 = -deltaPhi1*sizeWindowENeg_ ;
      float ephimax1 =  deltaPhi1*(1.-sizeWindowENeg_);
      float pphimin1 = -deltaPhi1*(1.-sizeWindowENeg_);
      float pphimax1 =  deltaPhi1*sizeWindowENeg_;
      myMatchEle->set1stLayer(ephimin1,ephimax1);
      myMatchPos->set1stLayer(pphimin1,pphimax1);
      myMatchEle->set2ndLayer(phimin2_,phimax2_);
      myMatchPos->set2ndLayer(phimin2_,phimax2_);

    }

  PropagationDirection dir = alongMomentum;
  
   // is this an electron
  double aCharge=-1.;
 
  std::vector<std::pair<RecHitWithDist,ConstRecHitPointer> > elePixelHits = 
  myMatchEle->compatibleHits(clusterPos,vertexPos, clusterEnergy, aCharge);
 
  float vertexZ = myMatchEle->getVertex();
  GlobalPoint eleVertex(BSPosition_.x(),BSPosition_.y(),vertexZ);
  int isEle = 0;
  if (!elePixelHits.empty() ) {
    LogDebug("") << "[ElectronPixelSeedGenerator::seedsFromThisCluster] electron compatible hits found ";
    isEle = 1;

    std::vector<std::pair<RecHitWithDist,ConstRecHitPointer> >::iterator v;
     
    for (v = elePixelHits.begin(); v != elePixelHits.end(); v++) {
      (*v).first.invert();
      
      bool valid = prepareElTrackSeed((*v).first.recHit(),(*v).second,eleVertex);
      if (valid) {

	double pOq =  (pts_->parameters()).mixedFormatVector()[0];
	
	if(pOq>0.)
	  {

	    double sign = (pts_->parameters()).pzSign();

	    double theDxdz = (pts_->parameters()).mixedFormatVector()[1];
	    double theDydz = (pts_->parameters()).mixedFormatVector()[2];  
	    double theX = (pts_->parameters()).mixedFormatVector()[3];  
	    double theY = (pts_->parameters()).mixedFormatVector()[4];

	    const  std::vector<float>  errmatrix = pts_->errorMatrix();
	    float myErrMatrix[15];

	    //  std::cout<<"\n matrixSize "<<errmatrix.size()<<std::endl;
	    for(int it=0; it<15; it++)
	      {
		myErrMatrix[it] = errmatrix[it]; 
	      }
	    const unsigned int id = pts_->detId();
	    const int surfSide = pts_->surfaceSide();

	    AlgebraicVector5 parameters;
	    parameters[0] = -pOq;
	    parameters[1] = theDxdz;
	    parameters[2] = theDydz;
	    parameters[3] = theX;
	    parameters[4] = theY;

	   const  LocalTrajectoryParameters seedParameters(parameters, sign, true);
	    
	   pts_ = new PTrajectoryStateOnDet(seedParameters,myErrMatrix, id,surfSide);
	  }
	
	reco::ElectronPixelSeed s(seedCluster,*pts_,recHits_,dir);

	//	std::cout<<"\n s charge "<<s.getCharge()<<std::endl;

	result.push_back(s);
	delete pts_;
	pts_=0;
      }
    }
  } 


  //try charge =1.
  aCharge=1.;  
  
  std::vector<std::pair<RecHitWithDist,ConstRecHitPointer> > posPixelHits = 
  myMatchPos->compatibleHits(clusterPos,vertexPos, clusterEnergy, aCharge);
 
  vertexZ = myMatchPos->getVertex();
  GlobalPoint posVertex(BSPosition_.x(),BSPosition_.y(),vertexZ);

  if (!posPixelHits.empty() ) {
    LogDebug("") << "[ElectronPixelSeedGenerator::seedsFromThisCluster] positron compatible hits found ";
    isEle == 1 ? isEle = 3 : isEle = 2;
    std::vector<std::pair<RecHitWithDist,ConstRecHitPointer> >::iterator v;
    for (v = posPixelHits.begin(); v != posPixelHits.end(); v++) {

      bool valid = prepareElTrackSeed((*v).first.recHit(),(*v).second,posVertex);
      if (valid) {
	double pOq =  (pts_->parameters()).mixedFormatVector()[0];
	if(pOq<0.)
	  {
	    double sign = (pts_->parameters()).pzSign();

	    double theDxdz = (pts_->parameters()).mixedFormatVector()[1];
	    double theDydz = (pts_->parameters()).mixedFormatVector()[2];  
	    double theX = (pts_->parameters()).mixedFormatVector()[3];  
	    double theY = (pts_->parameters()).mixedFormatVector()[4];

	    const  std::vector<float>  errmatrix = pts_->errorMatrix();
	    float myErrMatrix[15];

	    for(int it=0; it<15; it++)
	      {
		myErrMatrix[it] = errmatrix[it]; 
	      }
	    const unsigned int id = pts_->detId();
	    const int surfSide = pts_->surfaceSide();

	    AlgebraicVector5 parameters;
	    parameters[0] = -pOq;
	    parameters[1] = theDxdz;
	    parameters[2] = theDydz;
	    parameters[3] = theX;
	    parameters[4] = theY;

	    const  LocalTrajectoryParameters seedParameters(parameters, sign, true);
	    
	    pts_ = new PTrajectoryStateOnDet(seedParameters,myErrMatrix, id,surfSide);
	  }
	
	reco::ElectronPixelSeed s(seedCluster,*pts_,recHits_,dir);
	
	result.push_back(s);
	delete pts_;
	pts_=0;
      }
    }
  } 
  return ;
}

bool ElectronPixelSeedGenerator::prepareElTrackSeed(ConstRecHitPointer innerhit,
						    ConstRecHitPointer outerhit,
						    const GlobalPoint& vertexPos)
{
  
  // debug prints
  LogDebug("") <<"[ElectronPixelSeedGenerator::prepareElTrackSeed] inner PixelHit   x,y,z "<<innerhit->globalPosition();
  LogDebug("") <<"[ElectronPixelSeedGenerator::prepareElTrackSeed] outer PixelHit   x,y,z "<<outerhit->globalPosition();

  pts_=0;
  recHits_.clear();
  

  /*  
  SiPixelRecHit *hit;
  hit=new SiPixelRecHit(*(dynamic_cast <const SiPixelRecHit *> (innerhit->hit())));
  recHits_.push_back(hit);
  hit=new SiPixelRecHit(*(dynamic_cast <const SiPixelRecHit *> (outerhit->hit())));
  recHits_.push_back(hit);  
  */
  ///////////////
  
  SiPixelRecHit *pixhit=0;
  SiStripMatchedRecHit2D *striphit=0;
  const SiPixelRecHit* constpixhit = dynamic_cast <const SiPixelRecHit*> (innerhit->hit());
  if (constpixhit) {
    pixhit=new SiPixelRecHit(*constpixhit);
    recHits_.push_back(pixhit); 
  } else  return false;
  constpixhit =  dynamic_cast <const SiPixelRecHit *> (outerhit->hit());
  if (constpixhit) {
    pixhit=new SiPixelRecHit(*constpixhit);
    recHits_.push_back(pixhit); 
  } else {
    const SiStripMatchedRecHit2D * conststriphit=dynamic_cast <const SiStripMatchedRecHit2D *> (outerhit->hit());
    if (conststriphit) {
      striphit = new SiStripMatchedRecHit2D(*conststriphit);
      recHits_.push_back(striphit);   
    } else return false;
  }

  typedef TrajectoryStateOnSurface     TSOS;
  // make a spiral
  FastHelix helix(outerhit->globalPosition(),innerhit->globalPosition(),vertexPos,*theSetup);
  if ( !helix.isValid()) {
    return false;
  }
  FreeTrajectoryState fts = helix.stateAtVertex();
  TSOS propagatedState = thePropagator->propagate(fts,innerhit->det()->surface()) ;
  if (!propagatedState.isValid()) 
    return false;
  TSOS updatedState = theUpdator->update(propagatedState, *innerhit);
  
  TSOS propagatedState_out = thePropagator->propagate(fts,outerhit->det()->surface()) ;
  if (!propagatedState_out.isValid()) 
    return false;
  TSOS updatedState_out = theUpdator->update(propagatedState_out, *outerhit);
  // debug prints
  LogDebug("") <<"[ElectronPixelSeedGenerator::prepareElTrackSeed] final TSOS, position: "<<updatedState_out.globalPosition()<<" momentum: "<<updatedState_out.globalMomentum();
  LogDebug("") <<"[ElectronPixelSeedGenerator::prepareElTrackSeed] final TSOS Pt: "<<updatedState_out.globalMomentum().perp();
  pts_ =  transformer_.persistentState(updatedState_out, outerhit->geographicalId().rawId());

  return true;
}

///////////////////////////////       Energy correction factor for showering - golden/big brem/narrow "superclusters"
//float Ecorrection_Sh_GBbN(int numScl_matching)

float ElectronPixelSeedGenerator::fEtaBarrelBad(float scEta)
{
  // f(eta) for the class = 30 (estimated from 1Mevt single e sample)
  // Ivica's new corrections 01/06
  float p0 =  9.99063e-01;
  float p1 = -2.63341e-02;
  float p2 =  5.16054e-02;
  float p3 = -4.95976e-02;
  float p4 =  3.62304e-03;

  float x  = (float) fabs(scEta);
  return p0 + p1*x + p2*x*x + p3*x*x*x + p4*x*x*x*x;
}

float ElectronPixelSeedGenerator::fEtaEndcapGood(float scEta)
{
  // f(eta) for the first 3 classes (100, 110 and 120)
  // Ivica's new corrections 01/06
  float p0 =        -8.51093e-01;
  float p1 =         3.54266e+00;
  float p2 =        -2.59288e+00;
  float p3 =         8.58945e-01;
  float p4 =        -1.07844e-01;

  float x  = (float) fabs(scEta);
  return p0 + p1*x + p2*x*x + p3*x*x*x + p4*x*x*x*x;
}

float ElectronPixelSeedGenerator::fEtaEndcapBad(float scEta)
{
  // f(eta) for the class = 130-134
  // Ivica's new corrections 01/06
  float p0 =        -4.25221e+00;
  float p1 =         1.01936e+01;
  float p2 =        -7.48247e+00;
  float p3 =         2.45520e+00;
  float p4 =        -3.02872e-01;

  float x  = (float) fabs(scEta);
  return p0 + p1*x + p2*x*x + p3*x*x*x + p4*x*x*x*x;
}

float ElectronPixelSeedGenerator::fEtaBarrelGood(float scEta)
{
  // f(eta) for the first 3 classes (0, 10 and 20) (estimated from 1Mevt single e sample)
  // Ivica's new corrections 01/06
  float p0 =  1.00149e+00;
  float p1 = -2.06622e-03;
  float p2 = -1.08793e-02;
  float p3 =  1.54392e-02;
  float p4 = -1.02056e-02;

  float x  = (float) fabs(scEta);
  return p0 + p1*x + p2*x*x + p3*x*x*x + p4*x*x*x*x;
}
