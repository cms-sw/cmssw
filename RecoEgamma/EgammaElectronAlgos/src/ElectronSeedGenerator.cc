// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      ElectronSeedGenerator.
//
/**\class ElectronSeedGenerator EgammaElectronAlgos/ElectronSeedGenerator

 Description: Top algorithm producing ElectronSeeds, ported from ORCA

 Implementation:
     future redesign...
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
//

#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronSeedGenerator.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronUtilities.h"

#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"

//#include "DataFormats/EgammaReco/interface/EcalCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
//#include "DataFormats/BeamSpot/interface/BeamSpot.h"
//#include "DataFormats/Common/interface/Handle.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"

#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include <vector>
#include <utility>

ElectronSeedGenerator::ElectronSeedGenerator(const edm::ParameterSet &pset,
				      const ElectronSeedGenerator::Tokens& ts)
 : dynamicphiroad_(pset.getParameter<bool>("dynamicPhiRoad")),
   fromTrackerSeeds_(pset.getParameter<bool>("fromTrackerSeeds")),
   useRecoVertex_(false),
   verticesTag_(ts.token_vtx),
   beamSpotTag_(ts.token_bs),
   lowPtThreshold_(pset.getParameter<double>("LowPtThreshold")),
   highPtThreshold_(pset.getParameter<double>("HighPtThreshold")),
   nSigmasDeltaZ1_(pset.getParameter<double>("nSigmasDeltaZ1")),
   deltaZ1WithVertex_(0.5),
   sizeWindowENeg_(pset.getParameter<double>("SizeWindowENeg")),
   deltaPhi1Low_(pset.getParameter<double>("DeltaPhi1Low")),
   deltaPhi1High_(pset.getParameter<double>("DeltaPhi1High")),
   deltaPhi1Coef1_(0.), deltaPhi1Coef2_(0.),
   myMatchEle(0), myMatchPos(0),
   thePropagator(0),
   theMeasurementTracker(0),
   theMeasurementTrackerEventTag(ts.token_measTrkEvt),
   theSetup(0), 
   cacheIDMagField_(0),/*cacheIDGeom_(0),*/cacheIDNavSchool_(0),cacheIDCkfComp_(0),cacheIDTrkGeom_(0)
{
  // so that deltaPhi1 = deltaPhi1Coef1_ + deltaPhi1Coef2_/clusterEnergyT
  if (dynamicphiroad_)
    {
      deltaPhi1Coef2_ = (deltaPhi1Low_-deltaPhi1High_)/(1./lowPtThreshold_-1./highPtThreshold_) ;
      deltaPhi1Coef1_ = deltaPhi1Low_ - deltaPhi1Coef2_/lowPtThreshold_ ;
    }
  
  theMeasurementTrackerName = pset.getParameter<std::string>("measurementTrackerName"); 
  
  // use of reco vertex
  useRecoVertex_ = pset.getParameter<bool>("useRecoVertex");
  deltaZ1WithVertex_ = pset.getParameter<double>("deltaZ1WithVertex");
  
  // new B/F configurables
  deltaPhi2B_ = pset.getParameter<double>("DeltaPhi2B") ;
  deltaPhi2F_ = pset.getParameter<double>("DeltaPhi2F") ;

  phiMin2B_ = pset.getParameter<double>("PhiMin2B") ;
  phiMin2F_ = pset.getParameter<double>("PhiMin2F") ;

  phiMax2B_ = pset.getParameter<double>("PhiMax2B") ;
  phiMax2F_ = pset.getParameter<double>("PhiMax2F") ;

  // Instantiate the pixel hit matchers
  myMatchEle = new PixelHitMatcher
   ( pset.getParameter<double>("ePhiMin1"),
		 pset.getParameter<double>("ePhiMax1"),
		 phiMin2B_, phiMax2B_, phiMin2F_, phiMax2F_,
     pset.getParameter<double>("z2MinB"),
     pset.getParameter<double>("z2MaxB"),
     pset.getParameter<double>("r2MinF"),
     pset.getParameter<double>("r2MaxF"),
     pset.getParameter<double>("rMinI"),
     pset.getParameter<double>("rMaxI"),
     pset.getParameter<bool>("searchInTIDTEC") ) ;

  myMatchPos = new PixelHitMatcher
   ( pset.getParameter<double>("pPhiMin1"),
		 pset.getParameter<double>("pPhiMax1"),
		 phiMin2B_, phiMax2B_, phiMin2F_, phiMax2F_,
     pset.getParameter<double>("z2MinB"),
     pset.getParameter<double>("z2MaxB"),
     pset.getParameter<double>("r2MinF"),
     pset.getParameter<double>("r2MaxF"),
     pset.getParameter<double>("rMinI"),
     pset.getParameter<double>("rMaxI"),
     pset.getParameter<bool>("searchInTIDTEC") ) ;

  theUpdator = new KFUpdator() ;
}

ElectronSeedGenerator::~ElectronSeedGenerator()
 {
  delete myMatchEle ;
  delete myMatchPos ;
  delete thePropagator ;
  delete theUpdator ;
 }

void ElectronSeedGenerator::setupES(const edm::EventSetup& setup) {

  // get records if necessary (called once per event)
  bool tochange=false;

  if (cacheIDMagField_!=setup.get<IdealMagneticFieldRecord>().cacheIdentifier()) {
    setup.get<IdealMagneticFieldRecord>().get(theMagField);
    cacheIDMagField_=setup.get<IdealMagneticFieldRecord>().cacheIdentifier();
    if (thePropagator) delete thePropagator;
    thePropagator = new PropagatorWithMaterial(alongMomentum,.000511,&(*theMagField));
    tochange=true;
  }

  if (!fromTrackerSeeds_ && cacheIDCkfComp_!=setup.get<CkfComponentsRecord>().cacheIdentifier()) {
    edm::ESHandle<MeasurementTracker> measurementTrackerHandle;
    setup.get<CkfComponentsRecord>().get(theMeasurementTrackerName,measurementTrackerHandle);
    cacheIDCkfComp_=setup.get<CkfComponentsRecord>().cacheIdentifier();
    theMeasurementTracker = measurementTrackerHandle.product();
    tochange=true;
  }

  //edm::ESHandle<TrackerGeometry> trackerGeometryHandle;
  if (cacheIDTrkGeom_!=setup.get<TrackerDigiGeometryRecord>().cacheIdentifier()) {
    cacheIDTrkGeom_=setup.get<TrackerDigiGeometryRecord>().cacheIdentifier();
    setup.get<TrackerDigiGeometryRecord>().get(theTrackerGeometry);
    tochange=true; //FIXME
  }

  if (tochange) {
    myMatchEle->setES(&(*theMagField),theMeasurementTracker,theTrackerGeometry.product());
    myMatchPos->setES(&(*theMagField),theMeasurementTracker,theTrackerGeometry.product());
  }

  if (cacheIDNavSchool_!=setup.get<NavigationSchoolRecord>().cacheIdentifier()) {
    edm::ESHandle<NavigationSchool> nav;
    setup.get<NavigationSchoolRecord>().get("SimpleNavigationSchool", nav);
    cacheIDNavSchool_=setup.get<NavigationSchoolRecord>().cacheIdentifier();
    theNavigationSchool = nav.product();
  }

//  if (cacheIDGeom_!=setup.get<TrackerRecoGeometryRecord>().cacheIdentifier()) {
//    setup.get<TrackerRecoGeometryRecord>().get( theGeomSearchTracker );
//    cacheIDGeom_=setup.get<TrackerRecoGeometryRecord>().cacheIdentifier();
//  }

}

void display_seed( const std::string & title1, const std::string & title2, const reco::ElectronSeed & seed, edm::ESHandle<TrackerGeometry> trackerGeometry )
 {
  const PTrajectoryStateOnDet & startingState = seed.startingState() ;
  const LocalTrajectoryParameters & parameters = startingState.parameters() ;
  std::cout<<title1
    <<" ("<<seed.subDet2()<<"/"<<seed.dRz2()<<"/"<<seed.dPhi2()<<")"
    <<" ("<<seed.direction()<<"/"<<startingState.detId()<<"/"<<startingState.surfaceSide()<<"/"<<parameters.charge()<<"/"<<parameters.position()<<"/"<<parameters.momentum()<<")"
    <<std::endl ;
 }

bool equivalent( const TrajectorySeed & s1, const TrajectorySeed & s2 )
 {
  if (s1.nHits()!=s2.nHits()) return false ;

  unsigned int nHits ;
  TrajectorySeed::range r1 = s1.recHits(), r2 = s2.recHits() ;
  TrajectorySeed::const_iterator i1, i2 ;
  for ( i1=r1.first, i2=r2.first, nHits=0 ; i1!=r1.second ; ++i1, ++i2, ++nHits )
   {
    if ( !i1->isValid() || !i2->isValid() ) return false ;
    if ( i1->geographicalId()!=i2->geographicalId() ) return false ;
    if ( ! ( i1->localPosition()==i2->localPosition() ) ) return false ;
   }

  return true ;
 }

void  ElectronSeedGenerator::run
 ( edm::Event & e, const edm::EventSetup & setup,
   const reco::SuperClusterRefVector & sclRefs, const std::vector<float> & hoe1s, const std::vector<float> & hoe2s,
   TrajectorySeedCollection * seeds, reco::ElectronSeedCollection & out )
 {
  theInitialSeedColl = seeds ;
//  bool duplicateTrajectorySeeds =false ;
//  unsigned int i,j ;
//  for (i=0;i<seeds->size();++i)
//    for (j=i+1;j<seeds->size();++j)
//     {
//      const TrajectorySeed & s1 =(*seeds)[i] ;
//      const TrajectorySeed & s2 =(*seeds)[j] ;
//      if ( equivalent(s1,s2) )
//       {
//        const PTrajectoryStateOnDet & ss1 = s1.startingState() ;
//        const LocalTrajectoryParameters & p1 = ss1.parameters() ;
//        const PTrajectoryStateOnDet & ss2 = s2.startingState() ;
//        const LocalTrajectoryParameters & p2 = ss2.parameters() ;
//        duplicateTrajectorySeeds = true ;
//        std::cout<<"Same hits for "
//          <<"\n  s["<<i<<"] ("<<s1.direction()<<"/"<<ss1.detId()<<"/"<<ss1.surfaceSide()<<"/"<<p1.charge()<<"/"<<p1.position()<<"/"<<p1.momentum()<<")"
//          <<"\n  s["<<j<<"] ("<<s2.direction()<<"/"<<ss2.detId()<<"/"<<ss2.surfaceSide()<<"/"<<p2.charge()<<"/"<<p2.position()<<"/"<<p2.momentum()<<")"
//          <<std::endl ;
//       }
//     }
//  if (duplicateTrajectorySeeds)
//   { edm::LogWarning("ElectronSeedGenerator|DuplicateTrajectorySeeds")<<"We see several identical trajectory seeds." ; }

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHand;
  setup.get<TrackerTopologyRcd>().get(tTopoHand);
  const TrackerTopology *tTopo=tTopoHand.product();

  theSetup= &setup;


  // Step A: set Event for the TrajectoryBuilder
  edm::Handle<MeasurementTrackerEvent> data;
  e.getByToken(theMeasurementTrackerEventTag, data);
  myMatchEle->setEvent(*data);
  myMatchPos->setEvent(*data);

  // get initial TrajectorySeeds if necessary
  //  if (fromTrackerSeeds_) e.getByToken(initialSeeds_, theInitialSeedColl);

  // get the beamspot from the Event:
  //e.getByType(theBeamSpot);
  e.getByToken(beamSpotTag_,theBeamSpot);

  // if required get the vertices
  if (useRecoVertex_) e.getByToken(verticesTag_,theVertices);

  if (!fromTrackerSeeds_)
   { throw cms::Exception("NotSupported") << "Here in ElectronSeedGenerator " << __FILE__ << ":" << __LINE__ << " I would like to do theMeasurementTracker->update(e); but that no longer makes sense.\n"; 
   }

  for  (unsigned int i=0;i<sclRefs.size();++i) {
    // Find the seeds
    recHits_.clear();

    LogDebug ("run") << "new cluster, calling seedsFromThisCluster";
    seedsFromThisCluster(sclRefs[i],hoe1s[i],hoe2s[i],out,tTopo);
  }

  LogDebug ("run") << ": For event "<<e.id();
  LogDebug ("run") <<"Nr of superclusters after filter: "<<sclRefs.size()
   <<", no. of ElectronSeeds found  = " << out.size();
}

void ElectronSeedGenerator::seedsFromThisCluster
( edm::Ref<reco::SuperClusterCollection> seedCluster,
  float hoe1, float hoe2,
  reco::ElectronSeedCollection & out, const TrackerTopology *tTopo )
{
  float clusterEnergy = seedCluster->energy() ;
  GlobalPoint clusterPos
    ( seedCluster->position().x(),
      seedCluster->position().y(),
      seedCluster->position().z() ) ;
  reco::ElectronSeed::CaloClusterRef caloCluster(seedCluster) ;

  //LogDebug("") << "[ElectronSeedGenerator::seedsFromThisCluster] new supercluster with energy: " << clusterEnergy ;
  //LogDebug("") << "[ElectronSeedGenerator::seedsFromThisCluster] and position: " << clusterPos ;

  if (dynamicphiroad_)
   {
    float clusterEnergyT = clusterEnergy / cosh( EleRelPoint(clusterPos,theBeamSpot->position()).eta() ) ;

    float deltaPhi1 ;
    if (clusterEnergyT < lowPtThreshold_)
     { deltaPhi1= deltaPhi1Low_ ; }
    else if (clusterEnergyT > highPtThreshold_)
     { deltaPhi1= deltaPhi1High_ ; }
    else
     { deltaPhi1 = deltaPhi1Coef1_ + deltaPhi1Coef2_/clusterEnergyT ; }

    float ephimin1 = -deltaPhi1*sizeWindowENeg_ ;
    float ephimax1 =  deltaPhi1*(1.-sizeWindowENeg_);
    float pphimin1 = -deltaPhi1*(1.-sizeWindowENeg_);
    float pphimax1 =  deltaPhi1*sizeWindowENeg_;

    float phimin2B  = -deltaPhi2B_/2. ;
    float phimax2B  =  deltaPhi2B_/2. ;
    float phimin2F  = -deltaPhi2F_/2. ;
    float phimax2F  =  deltaPhi2F_/2. ;


    myMatchEle->set1stLayer(ephimin1,ephimax1);
    myMatchPos->set1stLayer(pphimin1,pphimax1);
    myMatchEle->set2ndLayer(phimin2B,phimax2B, phimin2F,phimax2F);
    myMatchPos->set2ndLayer(phimin2B,phimax2B, phimin2F,phimax2F);
   }

  PropagationDirection dir = alongMomentum ;

  if (!useRecoVertex_) // here use the beam spot position
   {
    double sigmaZ=theBeamSpot->sigmaZ();
    double sigmaZ0Error=theBeamSpot->sigmaZ0Error();
    double sq=sqrt(sigmaZ*sigmaZ+sigmaZ0Error*sigmaZ0Error);
    double myZmin1=theBeamSpot->position().z()-nSigmasDeltaZ1_*sq;
    double myZmax1=theBeamSpot->position().z()+nSigmasDeltaZ1_*sq;

    GlobalPoint vertexPos ;
    ele_convert(theBeamSpot->position(),vertexPos) ;

    myMatchEle->set1stLayerZRange(myZmin1,myZmax1);
    myMatchPos->set1stLayerZRange(myZmin1,myZmax1);

    if (!fromTrackerSeeds_)
     {
      // try electron
      std::vector<std::pair<RecHitWithDist,ConstRecHitPointer> > elePixelHits
       = myMatchEle->compatibleHits(clusterPos,vertexPos,
				    clusterEnergy,-1., tTopo, *theNavigationSchool) ;
      GlobalPoint eleVertex(theBeamSpot->position().x(),theBeamSpot->position().y(),myMatchEle->getVertex()) ;
      seedsFromRecHits(elePixelHits,dir,eleVertex,caloCluster,out,false) ;
      // try positron
      std::vector<std::pair<RecHitWithDist,ConstRecHitPointer> > posPixelHits
	= myMatchPos->compatibleHits(clusterPos,vertexPos,clusterEnergy,1.,tTopo, *theNavigationSchool) ;
      GlobalPoint posVertex(theBeamSpot->position().x(),theBeamSpot->position().y(),myMatchPos->getVertex()) ;
      seedsFromRecHits(posPixelHits,dir,posVertex,caloCluster,out,true) ;
     }
    else
     {
      // try electron
      std::vector<SeedWithInfo> elePixelSeeds
       = myMatchEle->compatibleSeeds(theInitialSeedColl,clusterPos,vertexPos,clusterEnergy,-1.) ;
      seedsFromTrajectorySeeds(elePixelSeeds,caloCluster,hoe1,hoe2,out,false) ;
      // try positron
      std::vector<SeedWithInfo> posPixelSeeds
       = myMatchPos->compatibleSeeds(theInitialSeedColl,clusterPos,vertexPos,clusterEnergy,1.) ;
      seedsFromTrajectorySeeds(posPixelSeeds,caloCluster,hoe1,hoe2,out,true) ;
     }

   }
  else // here we use the reco vertices
   {

    myMatchEle->setUseRecoVertex(true) ; //Hit matchers need to know that the vertex is known
    myMatchPos->setUseRecoVertex(true) ;

    const  std::vector<reco::Vertex> * vtxCollection = theVertices.product() ;
    std::vector<reco::Vertex>::const_iterator vtxIter ;
    for (vtxIter = vtxCollection->begin(); vtxIter != vtxCollection->end() ; vtxIter++)
     {

      GlobalPoint vertexPos(vtxIter->position().x(),vtxIter->position().y(),vtxIter->position().z());
      double myZmin1, myZmax1 ;
      if (vertexPos.z()==theBeamSpot->position().z())
       { // in case vetex not found
        double sigmaZ=theBeamSpot->sigmaZ();
        double sigmaZ0Error=theBeamSpot->sigmaZ0Error();
        double sq=sqrt(sigmaZ*sigmaZ+sigmaZ0Error*sigmaZ0Error);
        myZmin1=theBeamSpot->position().z()-nSigmasDeltaZ1_*sq;
        myZmax1=theBeamSpot->position().z()+nSigmasDeltaZ1_*sq;
       }
      else
       { // a vertex has been recoed
        myZmin1=vtxIter->position().z()-deltaZ1WithVertex_;
        myZmax1=vtxIter->position().z()+deltaZ1WithVertex_;
       }

      myMatchEle->set1stLayerZRange(myZmin1,myZmax1);
      myMatchPos->set1stLayerZRange(myZmin1,myZmax1);

      if (!fromTrackerSeeds_)
       {
        // try electron
        std::vector<std::pair<RecHitWithDist,ConstRecHitPointer> > elePixelHits
	  = myMatchEle->compatibleHits(clusterPos,vertexPos,clusterEnergy,-1.,tTopo, *theNavigationSchool) ;
        seedsFromRecHits(elePixelHits,dir,vertexPos,caloCluster,out,false) ;
        // try positron
	      std::vector<std::pair<RecHitWithDist,ConstRecHitPointer> > posPixelHits
		= myMatchPos->compatibleHits(clusterPos,vertexPos,clusterEnergy,1.,tTopo, *theNavigationSchool) ;
        seedsFromRecHits(posPixelHits,dir,vertexPos,caloCluster,out,true) ;
       }
      else
       {
        // try electron
        std::vector<SeedWithInfo> elePixelSeeds
         = myMatchEle->compatibleSeeds(theInitialSeedColl,clusterPos,vertexPos,clusterEnergy,-1.) ;
        seedsFromTrajectorySeeds(elePixelSeeds,caloCluster,hoe1,hoe2,out,false) ;
        // try positron
        std::vector<SeedWithInfo> posPixelSeeds
         = myMatchPos->compatibleSeeds(theInitialSeedColl,clusterPos,vertexPos,clusterEnergy,1.) ;
        seedsFromTrajectorySeeds(posPixelSeeds,caloCluster,hoe1,hoe2,out,true) ;
       }
     }
   }

  return ;
 }

void ElectronSeedGenerator::seedsFromRecHits
 ( std::vector<std::pair<RecHitWithDist,ConstRecHitPointer> > & pixelHits,
   PropagationDirection & dir,
   const GlobalPoint & vertexPos, const reco::ElectronSeed::CaloClusterRef & cluster,
   reco::ElectronSeedCollection & out,
   bool positron )
 {
  if (!pixelHits.empty())
   { LogDebug("ElectronSeedGenerator") << "Compatible "<<(positron?"positron":"electron")<<" hits found." ; }

  std::vector<std::pair<RecHitWithDist,ConstRecHitPointer> >::iterator v ;
  for ( v = pixelHits.begin() ; v != pixelHits.end() ; v++ )
   {
    if (!positron)
     { (*v).first.invert() ; }
    if (!prepareElTrackSeed((*v).first.recHit(),(*v).second,vertexPos))
     { continue ; }
    reco::ElectronSeed seed(pts_,recHits_,dir) ;
    seed.setCaloCluster(cluster) ;
    addSeed(seed,0,positron,out) ;
   }
 }

void ElectronSeedGenerator::seedsFromTrajectorySeeds
 ( const std::vector<SeedWithInfo> & pixelSeeds,
   const reco::ElectronSeed::CaloClusterRef & cluster,
   float hoe1, float hoe2,
   reco::ElectronSeedCollection & out,
   bool positron )
 {
  if (!pixelSeeds.empty())
   { LogDebug("ElectronSeedGenerator") << "Compatible "<<(positron?"positron":"electron")<<" seeds found." ; }

  std::vector<SeedWithInfo>::const_iterator s;
  for ( s = pixelSeeds.begin() ; s != pixelSeeds.end() ; s++ )
   {
    reco::ElectronSeed seed(s->seed()) ;
    seed.setCaloCluster(cluster,s->hitsMask(),s->subDet2(),s->subDet1(),hoe1,hoe2) ;
    addSeed(seed,&*s,positron,out) ;
   }
 }

void ElectronSeedGenerator::addSeed
 ( reco::ElectronSeed & seed,
   const SeedWithInfo * info,
   bool positron,
   reco::ElectronSeedCollection & out )
 {
  if (!info)
   { out.push_back(seed) ; return ; }

  if (positron)
   { seed.setPosAttributes(info->dRz2(),info->dPhi2(),info->dRz1(),info->dPhi1()) ; }
  else
   { seed.setNegAttributes(info->dRz2(),info->dPhi2(),info->dRz1(),info->dPhi1()) ; }
  reco::ElectronSeedCollection::iterator resItr ;
  for ( resItr=out.begin() ; resItr!=out.end() ; ++resItr )
   {
    if ( (seed.caloCluster()==resItr->caloCluster()) &&
         (seed.hitsMask()==resItr->hitsMask()) &&
         equivalent(seed,*resItr) )
     {
      if (positron)
       {
        if ( resItr->dRz2Pos()==std::numeric_limits<float>::infinity() &&
             resItr->dRz2()!=std::numeric_limits<float>::infinity() )
         {
          resItr->setPosAttributes(info->dRz2(),info->dPhi2(),info->dRz1(),info->dPhi1()) ;
          seed.setNegAttributes(resItr->dRz2(),resItr->dPhi2(),resItr->dRz1(),resItr->dPhi1()) ;
          break ;
         }
        else
         {
          if ( resItr->dRz2Pos()!=std::numeric_limits<float>::infinity() )
           {
            if ( resItr->dRz2Pos()!=seed.dRz2Pos() )
             {
              edm::LogWarning("ElectronSeedGenerator|BadValue")
               <<"this similar old seed already has another dRz2Pos"
               <<"\nold seed mask/dRz2/dPhi2/dRz2Pos/dPhi2Pos: "<<(unsigned int)resItr->hitsMask()<<"/"<<resItr->dRz2()<<"/"<<resItr->dPhi2()<<"/"<<resItr->dRz2Pos()<<"/"<<resItr->dPhi2Pos()
               <<"\nnew seed mask/dRz2/dPhi2/dRz2Pos/dPhi2Pos: "<<(unsigned int)seed.hitsMask()<<"/"<<seed.dRz2()<<"/"<<seed.dPhi2()<<"/"<<seed.dRz2Pos()<<"/"<<seed.dPhi2Pos() ;
             }
//            else
//             {
//              edm::LogWarning("ElectronSeedGenerator|UnexpectedValue")
//               <<"this old seed already knows its dRz2Pos, we suspect duplicates in input trajectry seeds"
//               <<"\nold seed mask/dRz2/dPhi2/dRz2Pos/dPhi2Pos: "<<(unsigned int)resItr->hitsMask()<<"/"<<resItr->dRz2()<<"/"<<resItr->dPhi2()<<"/"<<resItr->dRz2Pos()<<"/"<<resItr->dPhi2Pos()
//               <<"\nnew seed mask/dRz2/dPhi2/dRz2Pos/dPhi2Pos: "<<(unsigned int)seed.hitsMask()<<"/"<<seed.dRz2()<<"/"<<seed.dPhi2()<<"/"<<seed.dRz2Pos()<<"/"<<seed.dPhi2Pos() ;
//             }
            }
//          if (resItr->dRz2()==std::numeric_limits<float>::infinity())
//           {
//            edm::LogWarning("ElectronSeedGenerator|BadValue")
//             <<"this old seed has no dRz2, we suspect duplicates in input trajectry seeds"
//             <<"\nold seed mask/dRz2/dPhi2/dRz2Pos/dPhi2Pos: "<<(unsigned int)resItr->hitsMask()<<"/"<<resItr->dRz2()<<"/"<<resItr->dPhi2()<<"/"<<resItr->dRz2Pos()<<"/"<<resItr->dPhi2Pos()
//             <<"\nnew seed mask/dRz2/dPhi2/dRz2Pos/dPhi2Pos: "<<(unsigned int)seed.hitsMask()<<"/"<<seed.dRz2()<<"/"<<seed.dPhi2()<<"/"<<seed.dRz2Pos()<<"/"<<seed.dPhi2Pos() ;
//           }
         }
       }
      else
       {
        if ( resItr->dRz2()==std::numeric_limits<float>::infinity()
          && resItr->dRz2Pos()!=std::numeric_limits<float>::infinity() )
         {
          resItr->setNegAttributes(info->dRz2(),info->dPhi2(),info->dRz1(),info->dPhi1()) ;
          seed.setPosAttributes(resItr->dRz2Pos(),resItr->dPhi2Pos(),resItr->dRz1Pos(),resItr->dPhi1Pos()) ;
          break ;
         }
        else
         {
          if ( resItr->dRz2()!=std::numeric_limits<float>::infinity() )
           {
            if (resItr->dRz2()!=seed.dRz2())
             {
              edm::LogWarning("ElectronSeedGenerator|BadValue")
               <<"this old seed already has another dRz2"
               <<"\nold seed mask/dRz2/dPhi2/dRz2Pos/dPhi2Pos: "<<(unsigned int)resItr->hitsMask()<<"/"<<resItr->dRz2()<<"/"<<resItr->dPhi2()<<"/"<<resItr->dRz2Pos()<<"/"<<resItr->dPhi2Pos()
               <<"\nnew seed mask/dRz2/dPhi2/dRz2Pos/dPhi2Pos: "<<(unsigned int)seed.hitsMask()<<"/"<<seed.dRz2()<<"/"<<seed.dPhi2()<<"/"<<seed.dRz2Pos()<<"/"<<seed.dPhi2Pos() ;
             }
    //        else
    //         {
    //          edm::LogWarning("ElectronSeedGenerator|UnexpectedValue")
    //           <<"this old seed already knows its dRz2, we suspect duplicates in input trajectry seeds"
    //           <<"\nold seed mask/dRz2/dPhi2/dRz2Pos/dPhi2Pos: "<<(unsigned int)resItr->hitsMask()<<"/"<<resItr->dRz2()<<"/"<<resItr->dPhi2()<<"/"<<resItr->dRz2Pos()<<"/"<<resItr->dPhi2Pos()
    //           <<"\nnew seed mask/dRz2/dPhi2/dRz2Pos/dPhi2Pos: "<<(unsigned int)seed.hitsMask()<<"/"<<seed.dRz2()<<"/"<<seed.dPhi2()<<"/"<<seed.dRz2Pos()<<"/"<<seed.dPhi2Pos() ;
    //          seed.setPosAttributes(resItr->dRz2Pos(),resItr->dPhi2Pos(),resItr->dRz1Pos(),resItr->dPhi1Pos()) ;
    //         }
           }
//          if (resItr->dRz2Pos()==std::numeric_limits<float>::infinity())
//           {
//            edm::LogWarning("ElectronSeedGenerator|BadValue")
//             <<"this old seed has no dRz2Pos"
//             <<"\nold seed mask/dRz2/dPhi2/dRz2Pos/dPhi2Pos: "<<(unsigned int)resItr->hitsMask()<<"/"<<resItr->dRz2()<<"/"<<resItr->dPhi2()<<"/"<<resItr->dRz2Pos()<<"/"<<resItr->dPhi2Pos()
//             <<"\nnew seed mask/dRz2/dPhi2/dRz2Pos/dPhi2Pos: "<<(unsigned int)seed.hitsMask()<<"/"<<seed.dRz2()<<"/"<<seed.dPhi2()<<"/"<<seed.dRz2Pos()<<"/"<<seed.dPhi2Pos() ;
//           }
         }
       }
     }
   }

  out.push_back(seed) ;
 }

bool ElectronSeedGenerator::prepareElTrackSeed
 ( ConstRecHitPointer innerhit,
   ConstRecHitPointer outerhit,
   const GlobalPoint& vertexPos )
 {

  // debug prints
  LogDebug("") <<"[ElectronSeedGenerator::prepareElTrackSeed] inner PixelHit   x,y,z "<<innerhit->globalPosition();
  LogDebug("") <<"[ElectronSeedGenerator::prepareElTrackSeed] outer PixelHit   x,y,z "<<outerhit->globalPosition();

  recHits_.clear();

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

  // FIXME to be optimized outside the loop
  edm::ESHandle<MagneticField> bfield;
  theSetup->get<IdealMagneticFieldRecord>().get(bfield);
  float nomField = bfield->nominalValue();

  // make a spiral
  FastHelix helix(outerhit->globalPosition(),innerhit->globalPosition(),vertexPos,nomField,&*bfield);
  if ( !helix.isValid()) {
    return false;
  }
  FreeTrajectoryState fts(helix.stateAtVertex());
  TSOS propagatedState = thePropagator->propagate(fts,innerhit->det()->surface()) ;
  if (!propagatedState.isValid())
    return false;
  TSOS updatedState = theUpdator->update(propagatedState, *innerhit);

  TSOS propagatedState_out = thePropagator->propagate(updatedState,outerhit->det()->surface()) ;
  if (!propagatedState_out.isValid())
    return false;
  TSOS updatedState_out = theUpdator->update(propagatedState_out, *outerhit);
  // debug prints
  LogDebug("") <<"[ElectronSeedGenerator::prepareElTrackSeed] final TSOS, position: "<<updatedState_out.globalPosition()<<" momentum: "<<updatedState_out.globalMomentum();
  LogDebug("") <<"[ElectronSeedGenerator::prepareElTrackSeed] final TSOS Pt: "<<updatedState_out.globalMomentum().perp();
  pts_ =  trajectoryStateTransform::persistentState(updatedState_out, outerhit->geographicalId().rawId());

  return true;
 }
