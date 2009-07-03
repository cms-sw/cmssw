// -*- C++ -*-
//
// Package:    TestMuL1L2
// Class:      TestMuL1L2
//
// \class TestMuL1L2 TestMuL1L2.cc TestMuonL1L2/TestMuL1L2/src/TestMuL1L2.cc
//
// Original Author:  Dong Ho Moon
//         Created:  Wed May  9 06:22:36 CEST 2007
// $Id: HITrackVertexMaker.cc,v 1.13 2009/07/02 16:37:06 kodolova Exp $
//
//
 
#include "RecoHIMuon/HiMuTracking/interface/HITrackVertexMaker.h" 

// C++ Headers

#include <memory>
#include<iostream>
#include<iomanip>
#include<vector>
#include<cmath>

// ES include files 
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

// Muon trigger includes 
#include "L1Trigger/GlobalMuonTrigger/interface/L1MuGlobalMuonTrigger.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutRecord.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

// Tracking includes
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/GeomPropagators/interface/StateOnTrackerBound.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryFiltering/interface/RegionalTrajectoryFilter.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "RecoTracker/TkNavigation/interface/NavigationSchoolFactory.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
//#include "TrackingTools/PatternTools/interface/TrajectoryStateClosestToBeamLineBuilder.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "RecoTracker/TkNavigation/interface/SimpleNavigationSchool.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "TrackingTools/PatternTools/interface/TrajectorySmoother.h"
#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
// Geometry includes

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

// RecoHIMuon includes

#include "RecoHIMuon/HiMuSeed/interface/HICFTSfromL1orL2.h"
#include "RecoHIMuon/HiMuSeed/interface/HICConst.h"
#include "RecoHIMuon/HiMuPropagator/interface/FastMuPropagator.h"
#include "RecoHIMuon/HiMuPropagator/interface/FmpConst.h"
#include "RecoHIMuon/HiMuPropagator/interface/HICTkOuterStartingLayerFinder.h"
#include "RecoHIMuon/HiMuSeed/interface/DiMuonSeedGeneratorHIC.h"
#include "RecoHIMuon/HiMuTracking/interface/HICTrajectoryBuilder.h"
#include "RecoHIMuon/HiMuTracking/interface/HICSimpleNavigationSchool.h"
#include "RecoHIMuon/HiMuTracking/interface/HICMuonUpdator.h"
#include "RecoHIMuon/HiMuSeed/interface/HICConst.h"

// Global  points

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"


//Constructor

using namespace reco;
using namespace std;

//#define DEBUG

namespace cms{
HITrackVertexMaker::HITrackVertexMaker(const edm::ParameterSet& ps1, const edm::EventSetup& es1)
{

   L2candTag_          = ps1.getParameter< edm::InputTag > ("L2CandTag");
   rphirecHitsTag      = ps1.getParameter< edm::InputTag > ("rphiRecHits");
   builderName         = ps1.getParameter< std::string >   ("TTRHBuilder");
   primaryVertexTag    = ps1.getParameter< edm::InputTag > ("PrimaryVertexTag");

#ifdef DEBUG
   std::cout<<" Start HI TrackVertexMaker constructor "<<std::endl;
#endif
   pset_ = ps1;

//
// Initializetion from Records
    es1.get<TrackerRecoGeometryRecord>().get( tracker );
    es1.get<IdealMagneticFieldRecord>().get(magfield);
    es1.get<CkfComponentsRecord>().get("",measurementTrackerHandle);
    es1.get<TransientRecHitRecord>().get(builderName,recHitBuilderHandle); 
       
    double theChiSquareCut = 500.;
    double nsig = 3.;
//    int theLowMult = 1;
    theEstimator = new HICMeasurementEstimator(&(*tracker), &(*magfield), theChiSquareCut, nsig);
    std::string updatorName = "KFUpdator";
    std::string propagatorAlongName    = "PropagatorWithMaterial";
    std::string propagatorOppositeName = "PropagatorWithMaterialOpposite";
    es1.get<TrackingComponentsRecord>().get(propagatorAlongName,propagatorAlongHandle);
    es1.get<TrackingComponentsRecord>().get(propagatorOppositeName,propagatorOppositeHandle);
    es1.get<TrackingComponentsRecord>().get(updatorName,updatorHandle);
    double ptmin=1.;
    theMinPtFilter = new MinPtTrajectoryFilter(ptmin); 
    theTrajectoryBuilder = new HICTrajectoryBuilder(        pset_, es1,
                                                     updatorHandle.product(),
                                                     propagatorAlongHandle.product(),
                                                     propagatorOppositeHandle.product(),
                                                     theEstimator,
                                                     recHitBuilderHandle.product(),
                                                     measurementTrackerHandle.product(),
                                                     theMinPtFilter);
// Fill vector of navigation schools for barrel
   int lost=-1;
   int lostf=-1;
   theNavigationSchoolV.push_back(new HICSimpleNavigationSchool(&(*tracker), &(*magfield), lost, lostf));  
   for(int i=11;i>-1;i--) {
     theNavigationSchoolV.push_back(new HICSimpleNavigationSchool(&(*tracker), &(*magfield), i, lostf));
   } 
   lost=-1;    
   for(int i=12;i>-1;i--) {
     theNavigationSchoolV.push_back(new HICSimpleNavigationSchool(&(*tracker), &(*magfield), lost, i));
   }
#ifdef DEBUG
    std::cout<<" HICTrajectoryBuilder constructed "<<std::endl;
#endif
}



//Destructor

HITrackVertexMaker::~HITrackVertexMaker()
{
} 

bool HITrackVertexMaker::produceTracks(const edm::Event& e1, const edm::EventSetup& es1, HICConst* theHICConst, FmpConst* theFmpConst)
{
    
   bool dimuon = false;
   edm::Handle<reco::VertexCollection> vertexcands;
   e1.getByLabel (primaryVertexTag,vertexcands);
   int iv = 0;

#ifdef DEBUG   
   cout<<" Number of vertices primary  "<<vertexcands->size()<<endl;
   if(vertexcands->size()<1) cout<<" Primary vertex failed "<<endl;
#endif

   if(vertexcands->size()<1) cout<<" Primary vertex failed "<<endl;

   if(vertexcands->size()<1) return dimuon;

   for (reco::VertexCollection::const_iterator ipvertex=vertexcands->begin();ipvertex!=vertexcands->end();ipvertex++)
   {
//     cout<<" Vertex position from pixels "<<(*ipvertex).position().z()<<endl;
     if (iv == 0) {theHICConst->setVertex((*ipvertex).position().z()); theFmpConst->setVertex((*ipvertex).position().z());} 
     iv++;
   } 

//   cout << " Vertex is set to (found by pixel finder)"<<theHICConst->zvert<<endl;
 
//
// Get recHit builder
//

  es1.get<TransientRecHitRecord>().get(builderName,recHitBuilderHandle);

//
// Get measurement tracker
//
//  
 
  es1.get<CkfComponentsRecord>().get("",measurementTrackerHandle);
    
#ifdef DEBUG  
  std::cout<<" Before first tracker update "<<std::endl;
#endif

  measurementTrackerHandle->update(e1);
  
#ifdef DEBUG
  std::cout<<" After first tracker update "<<std::endl;
#endif

//   edm::Handle<RecoChargedCandidateCollection> L2mucands;
   edm::Handle<TrackCollection> L2mucands;
   e1.getByLabel (L2candTag_,L2mucands);
//   RecoChargedCandidateCollection::const_iterator L2cand1;
//   RecoChargedCandidateCollection::const_iterator L2cand2;
      
#ifdef DEBUG
   cout<<" Number of muon candidates "<<L2mucands->size()<<endl;
   if( L2mucands->size() < 2 ) cout<<"L2 muon failed"<<endl;
#endif

   if( L2mucands->size() < 2 ) return dimuon;

#ifdef DEBUG
   cout<<"L2 muon accepted"<<endl;
#endif

// For trajectory builder   
   int  theLowMult = 1;
   theEstimator->setHICConst(theHICConst);
   theEstimator->setMult(theLowMult);
//   int lost=-1;
//   theNavigationSchool = new HICSimpleNavigationSchool(&(*tracker), &(*magfield), lost);
//   lost=10;
//   theNavigationSchooln = new HICSimpleNavigationSchool(&(*tracker), &(*magfield), lost);
   theTrajectoryBuilder->settracker(measurementTrackerHandle.product());
//============================   
    FastMuPropagator* theFmp = new FastMuPropagator(&(*magfield),theFmpConst); 
    StateOnTrackerBound state(theFmp); 
    TrajectoryStateOnSurface tsos;
    
#ifdef DEBUG   
//   for (cand1=L2mucands->begin(); cand1!=L2mucands->end(); cand1++) {
//     reco::TrackRef mytr = (*cand1).track(); 
//      std::cout<<" Inner position "<<(*mytr).innerPosition().x()<<" "<<(*mytr).innerPosition().y()<<" "<<(*mytr).innerPosition().z()<<std::endl;
//
//   } 
#endif 
  
    HICFTSfromL1orL2 vFts(&(*magfield));
    
    
    int NumOfSigma=4;
    HICTkOuterStartingLayerFinder TkOSLF(NumOfSigma, &(*magfield), &(*tracker), theHICConst);
   
    int mult = 1;
    DiMuonSeedGeneratorHIC Seed(rphirecHitsTag,&(*magfield),&(*tracker), theHICConst, builderName, mult);

//    vector<FreeTrajectoryState> theFts = vFts.createFTSfromStandAlone((*mucands));

   vector<FreeTrajectoryState> theFts = vFts.createFTSfromL2((*L2mucands)); 

#ifdef DEBUG
    cout<<" Size of the freeTS "<<theFts.size()<<endl;
#endif 
   
   
    
   DiMuonSeedGeneratorHIC::SeedContainer myseeds;  
   map<DetLayer*, DiMuonSeedGeneratorHIC::SeedContainer> seedmap;
   vector<Trajectory> theTmpTrajectories0;
 
//   map< FreeTrajectoryState*, Trajectory>  

   vector<FreeTrajectoryState*> theFoundFts;
   map<FreeTrajectoryState*, vector<Trajectory> >  theMapFtsTraj;


   for(vector<FreeTrajectoryState>::iterator ifts=theFts.begin(); ifts!=theFts.end(); ifts++)
   {
      theTmpTrajectories0.clear();
#ifdef DEBUG
     cout<<" cycle on Muon Trajectory State " <<(*ifts).parameters().position().perp()<<
                                          " " <<(*ifts).parameters().position().z()   <<endl;
#endif

     tsos=state((*ifts));          
     if(tsos.isValid())
     {

//        vector<Trajectory> theTmpTrajectories0;
#ifdef DEBUG
        cout<<" Position "<<tsos.globalPosition().perp()<<" "<<tsos.globalPosition().phi()<<
	      " "<<tsos.globalPosition().z()<<" "<<tsos.globalMomentum().perp()<<endl;
#endif

// Start to find starting layers
	FreeTrajectoryState* ftsnew=tsos.freeTrajectoryState();
        vector<DetLayer*> seedlayers = TkOSLF.startingLayers((*ftsnew));
	
#ifdef DEBUG
	std::cout<<" The size of the starting layers "<<seedlayers.size()<<std::endl;
#endif

	if( seedlayers.size() == 0 ) {
#ifdef DEBUG
          cout<<" Starting layers failed for muon candidate "<<endl;
#endif
          continue;
        }
        map<DetLayer*, DiMuonSeedGeneratorHIC::SeedContainer> seedmap = Seed.produce(e1 ,es1, 
                                                          (*ftsnew), tsos, (*ifts), 
	                                                   recHitBuilderHandle.product(),
                                                           measurementTrackerHandle.product(), 
                                                           &seedlayers);

 

       for( vector<DetLayer*>::iterator it = seedlayers.begin(); it != seedlayers.end(); it++)
       {

       DiMuonSeedGeneratorHIC::SeedContainer seeds = (*seedmap.find(*it)).second;

#ifdef DEBUG
       std::cout<<" Layer::Position z "<<(**it).surface().position().z()<<
                  " Number of seeds in layer "<<seeds.size()<<std::endl;	    
#endif

  if(seeds.size() == 0) {
#ifdef DEBUG
    std::cout<<" No seeds are found: do not continue with this Detlayer "<<std::endl;   
#endif
    continue;
  }
    // set the first navigation (all layers without gap)
       
       NavigationSetter setter( *(theNavigationSchoolV[0]));
    
       for(DiMuonSeedGeneratorHIC::SeedContainer::iterator iseed=seeds.begin();
                                                           iseed!=seeds.end();iseed++)
       {
         std::vector<TrajectoryMeasurement> theV = (*iseed).measurements();

#ifdef DEBUG
         std::cout<< "RecHIT Layer position r "<<
         theV[0].recHit()->globalPosition().perp()<<" phi "<<
         theV[0].recHit()->globalPosition().phi()<<" z "<<
         theV[0].recHit()->globalPosition().z()<<" momentum "<<
         theV[0].updatedState().freeTrajectoryState()->parameters().momentum().perp()<<" "<<
         theV[0].updatedState().freeTrajectoryState()->parameters().momentum().z()<<std::endl;
#endif

	 vector<Trajectory> theTmpTrajectories = theTrajectoryBuilder->trajectories(*iseed); 
    
#ifdef DEBUG
        cout<<" Number of found trajectories "<<theTmpTrajectories.size()<<endl;	 
#endif
        if(theTmpTrajectories.size()>0) {
//	allTraj.insert(allTraj.end(),theTmpTrajectories.begin(),theTmpTrajectories.end());
        theTmpTrajectories0.insert(theTmpTrajectories0.end(),
                                   theTmpTrajectories.begin(),
                                   theTmpTrajectories.end());
#ifdef DEBUG
        std::cout<<"We found trajectories for at least one seed "<<theTmpTrajectories0.size()<<std::endl; 
#endif
        break;
        }  // There are trajectories
       } // seeds 	    

    if( theTmpTrajectories0.size() > 0 ) {
#ifdef DEBUG
        std::cout<<"We found trajectories for at least one seed "<<theTmpTrajectories0.size()<<"break layer cycle "<<std::endl; 
#endif
     break;  
    }
   } // seedlayer



    if( theTmpTrajectories0.size() > 0 ) {
#ifdef DEBUG
        std::cout<<"We found trajectories for at least one seed "<<theTmpTrajectories0.size()
                 <<"continue"<<std::endl; 
#endif
     theFoundFts.push_back(&(*ifts));
     theMapFtsTraj[&(*ifts)] = theTmpTrajectories0;
     continue;
    } 

#ifdef DEBUG
        std::cout<<"No trajectories for this FTS "<<theTmpTrajectories0.size()<<
                   "try second path"<<std::endl; 
#endif

// No trajectories found for this Muon FTS although seeds exist. 
// Try to allow another trajectory map with lost 1 layer (only for barrel layers)

       for( vector<DetLayer*>::iterator it = seedlayers.begin(); it != seedlayers.end(); it++)
       {

          DiMuonSeedGeneratorHIC::SeedContainer seeds = (*seedmap.find(*it)).second;

          if(seeds.size() == 0) {
#ifdef DEBUG
    std::cout<<" Second path: No seeds are found: do not continue with this Detlayer "<<std::endl;   
#endif
             continue;
           }

       if((**it).location() == GeomDetEnumerators::endcap) {
       for(unsigned int j=13; j<theNavigationSchoolV.size();j++){
       NavigationSetter setter( *(theNavigationSchoolV[j]));
       for(DiMuonSeedGeneratorHIC::SeedContainer::iterator iseed=seeds.begin();
                                                           iseed!=seeds.end();iseed++)
       {
         std::vector<TrajectoryMeasurement> theV = (*iseed).measurements();
         vector<Trajectory> theTmpTrajectories = theTrajectoryBuilder->trajectories(*iseed);

         if(theTmpTrajectories.size()>0) {
//	 allTraj.insert(allTraj.end(),theTmpTrajectories.begin(),theTmpTrajectories.end());
         theTmpTrajectories0.insert(theTmpTrajectories0.end(),
                                    theTmpTrajectories.begin(),
                                    theTmpTrajectories.end());
#ifdef DEBUG
         std::cout<<"Second path: We found trajectories for at least one seed in barrel layer "<<
                                    theTmpTrajectories0.size()<<std::endl; 
#endif
          break;
         }  // There are trajectories
        } // seeds

         if( theTmpTrajectories0.size() > 0 ) {
#ifdef DEBUG
         std::cout<<"Second path: no trajectories: we try next barrel layer "<<
                                  theTmpTrajectories0.size()<<std::endl; 
#endif
                    break;
         }   
       } // navigation maps
       } else {
       for(int j=1; j<13;j++){
       NavigationSetter setter(*(theNavigationSchoolV[j]));
       for(DiMuonSeedGeneratorHIC::SeedContainer::iterator iseed=seeds.begin();
                                                           iseed!=seeds.end();iseed++)
       {
         std::vector<TrajectoryMeasurement> theV = (*iseed).measurements();
         vector<Trajectory> theTmpTrajectories = theTrajectoryBuilder->trajectories(*iseed);

         if(theTmpTrajectories.size()>0) {
//	 allTraj.insert(allTraj.end(),theTmpTrajectories.begin(),theTmpTrajectories.end());
         theTmpTrajectories0.insert(theTmpTrajectories0.end(),
                                    theTmpTrajectories.begin(),
                                    theTmpTrajectories.end());
#ifdef DEBUG
         std::cout<<"Second path: We found trajectories for at least one seed in barrel layer "<<
                                  theTmpTrajectories0.size()<<std::endl; 
#endif
          break;
         }  // There are trajectories
        } // seeds

         if( theTmpTrajectories0.size() > 0 ) {
#ifdef DEBUG
         std::cout<<
   "Second path:  We found trajectories for at least one seed in barrel/endcap layer"<<
                                  theTmpTrajectories0.size()<<std::endl; 
#endif
                    break;
         }   
     } // navigation maps
    } // barrel or endcap seed
         if( theTmpTrajectories0.size() > 0 ) {
#ifdef DEBUG
         std::cout<<
              "Second path: We found trajectories for at least one seed in barrel/endcap layer "
                               <<
                                  theTmpTrajectories0.size()<<std::endl; 
#endif     
              theFoundFts.push_back(&(*ifts));     
              theMapFtsTraj[&(*ifts)] = theTmpTrajectories0;
              break;
         }  
    } // seedlayer
   } // tsos. isvalid
 } // Muon Free trajectory state
        
//
// start fitting procedure
//
#ifdef DEBUG
    if(theFoundFts.size()>0 ) {
     std::cout<<" Event reconstruction finished with "<<theFoundFts.size()  
              <<std::endl;} 
           else {std::cout<<" Event reconstruction::no tracks found"<<std::endl;}
#endif
    if(theFoundFts.size()<2)  return dimuon;

// Look for vertex constraints

    edm::ESHandle<GlobalTrackingGeometry> globTkGeomHandle;
    es1.get<GlobalTrackingGeometryRecord>().get(globTkGeomHandle);

    reco::BeamSpot::CovarianceMatrix matrix;
    matrix(2,2) = 0.001;
    matrix(3,3) = 0.001;

    reco::BeamSpot bs( reco::BeamSpot::Point(0., 0., theHICConst->zvert),
                                                     0.1,
                                                     0.,
                                                     0.,
                                                     0.,
                                                    matrix
                     );


    reco::TrackBase::TrackAlgorithm Algo = reco::TrackBase::undefAlgorithm;

//    edm::ESHandle<TrajectoryFitter> theFitterTrack;
//    edm::ESHandle<TrajectorySmoother> theSmootherTrack;
//    edm::ESHandle<Propagator> thePropagatorTrack;
//    es1.get<TrackingComponentsRecord>().get("KFFitterForRefitInsideOut",theFitterTrack);
//    es1.get<TrackingComponentsRecord>().get("KFSmootherForRefitInsideOut",theSmootherTrack);
//    es1.get<TrackingComponentsRecord>().get("SmartPropagatorAny",thePropagatorTrack);
//    enum RefitDirection{insideOut,outsideIn,undetermined};

// For trajectory refitting
        vector<reco::Track> firstTrack;
        vector<reco::TransientTrack> firstTransTracks;
        vector<reco::TrackRef> firstTrackRefs;
        vector<reco::Track> secondTrack;
        vector<reco::TransientTrack> secondTransTracks;
        vector<reco::TrackRef> secondTrackRefs;

    for(vector<FreeTrajectoryState*>::iterator it = theFoundFts.begin(); it!= theFoundFts.end(); it++)
    {
        vector<Trajectory> first = (*theMapFtsTraj.find(*it)).second;

        for(vector<Trajectory>::iterator im=first.begin();im!=first.end(); im++) {

          TrajectoryStateOnSurface innertsos;
          if (im->direction() == alongMomentum) {
           innertsos = im->firstMeasurement().updatedState();
          } else {
            innertsos = im->lastMeasurement().updatedState();
          }


         // CMSSW31X
          TSCBLBuilderNoMaterial tscblBuilder;
         // CMSSW22X
         //TrajectoryStateClosestToBeamLineBuilder tscblBuilder;
         TrajectoryStateClosestToBeamLine tscbl = tscblBuilder(*(innertsos.freeState()),bs);

         if (tscbl.isValid()==false) {
            //cout<<" false track "<<endl; 
         continue;
         }

         GlobalPoint v = tscbl.trackStateAtPCA().position();
         math::XYZPoint  pos( v.x(), v.y(), v.z() );
    
//    cout<<" Position of track close to vertex "<<v.perp()<<" "<<v.z()<<" Primary vertex "<<theHICConst->zvert<<
//                                     " charge "<<tscbl.trackStateAtPCA().charge()<<endl;
   
         if(v.perp() > 0.1 ) continue;
         if(fabs(v.z() - theHICConst->zvert ) > 0.4 ) continue;

          GlobalVector p = tscbl.trackStateAtPCA().momentum();
          math::XYZVector mom( p.x(), p.y(), p.z() );

          Track theTrack(im->chiSquared(),
                         im->ndof(), 
                         pos, mom, tscbl.trackStateAtPCA().charge(), 
                         tscbl.trackStateAtPCA().curvilinearError(),Algo);
           TransientTrack tmpTk( theTrack, &(*magfield), globTkGeomHandle );

         
           firstTrack.push_back( theTrack );
           firstTransTracks.push_back( tmpTk );
        }

    if( firstTrack.size() == 0 ) continue;

    for(vector<FreeTrajectoryState*>::iterator jt = it+1; jt!= theFoundFts.end(); jt++)
    {
        vector<Trajectory> second = (*theMapFtsTraj.find(*jt)).second;
        cout<<" Number of trajectories first "<<first.size()<<" second "<<second.size()<<endl;

        for(vector<Trajectory>::iterator im=second.begin();im!=second.end(); im++) {

          TrajectoryStateOnSurface innertsos;

          if (im->direction() == alongMomentum) {
           innertsos = im->firstMeasurement().updatedState();
          } else {
            innertsos = im->lastMeasurement().updatedState();
          }

         TSCBLBuilderNoMaterial  tscblBuilder;
         TrajectoryStateClosestToBeamLine tscbl = tscblBuilder(*(innertsos.freeState()),bs);

         if (tscbl.isValid()==false) {
            //cout<<" false track "<<endl; 
         continue;
         }

         GlobalPoint v = tscbl.trackStateAtPCA().position();
         math::XYZPoint  pos( v.x(), v.y(), v.z() );
    
//    cout<<" Position of track close to vertex "<<v.perp()<<" "<<v.z()<<" Primary vertex "<<theHICConst->zvert<<
//                                     " charge "<<tscbl.trackStateAtPCA().charge()<<endl;
   
         if(v.perp() > 0.1 ) continue;
         if(fabs(v.z() - theHICConst->zvert ) > 0.4 ) continue;

          GlobalVector p = tscbl.trackStateAtPCA().momentum();
          math::XYZVector mom( p.x(), p.y(), p.z() );

          Track theTrack(im->chiSquared(),
                         im->ndof(), 
                         pos, mom, tscbl.trackStateAtPCA().charge(), 
                             tscbl.trackStateAtPCA().curvilinearError(),Algo);
           TransientTrack tmpTk( theTrack, &(*magfield), globTkGeomHandle );

         
           secondTrack.push_back( theTrack );
           secondTransTracks.push_back( tmpTk );
        }
        if( secondTrack.size() == 0 ) continue;
        cout<<" Number of tracks first "<<firstTrack.size()<<" second "<<secondTrack.size()<<endl;

    } // FTS
  } // FTS 


   if( firstTrack.size() < 1 || secondTrack.size() < 1 ){ 
#ifdef DEBUG
     cout<<" No enough tracks to get vertex "<<endl; 
#endif
     return dimuon; 
   }

   bool useRefTrax=true;
   KalmanVertexFitter theFitter(useRefTrax);
   TransientVertex theRecoVertex;
//
// Create possible two particles vertices
//
   vector<reco::TransientTrack> theTwoTransTracks;

//   vector<TransientVertex> theVertexContainer;

   for(vector<reco::TransientTrack>::iterator iplus = firstTransTracks.begin(); 
                                              iplus != firstTransTracks.end(); iplus++)
   {
     theTwoTransTracks.clear();
     theTwoTransTracks.push_back(*iplus);
     for(vector<reco::TransientTrack>::iterator iminus = secondTransTracks.begin(); 
                                                iminus != secondTransTracks.end(); iminus++)
     {
       theTwoTransTracks.push_back(*iminus);
       theRecoVertex = theFitter.vertex(theTwoTransTracks);
      if( !theRecoVertex.isValid() ) {
        continue;
      } 
      
//        cout<<" Vertex is found "<<endl;
//        cout<<" Chi2 = "<<theRecoVertex.totalChiSquared()<<
//	          " r= "<<theRecoVertex.position().perp()<<
//		  " z= "<<theRecoVertex.position().z()<<endl;

// Additional cuts       
     if ( theRecoVertex.totalChiSquared() > 0.0002 ) {
//        cout<<" Vertex is failed with Chi2 : "<<theRecoVertex.totalChiSquared()<<endl; 
     continue;
     }
     if ( theRecoVertex.position().perp() > 0.08 ) {
//        cout<<" Vertex is failed with r position : "<<theRecoVertex.position().perp()<<endl; 
	continue;
      }    
     if ( fabs(theRecoVertex.position().z()-theHICConst->zvert) > 0.06 ) {
//         cout<<" Vertex is failed with z position : "<<theRecoVertex.position().z()<<endl; 
	 continue;
     }
     double quality = theRecoVertex.normalisedChiSquared();
     std::vector<reco::TransientTrack> tracks = theRecoVertex.originalTracks();
     vector<TransientTrack> refittedTrax;
     if( theRecoVertex.hasRefittedTracks() ) {
          refittedTrax = theRecoVertex.refittedTracks();
     }

     for (std::vector<reco::TransientTrack>::iterator ivb = tracks.begin(); ivb != tracks.end(); ivb++)
     {
      quality = quality * (*ivb).chi2() /(*ivb).ndof();
     }
      if( quality > 70. ) {
            //    cout<<" Vertex failed quality cut "<<quality<<endl; 
		continue;
      }
//      theVertexContainer.push_back(theRecoVertex);
      
      dimuon = true;
      return dimuon;

     } // iminus
  } // iplus

/*  
       if( theVertexContainer.size() < 1 ) { 
#ifdef DEBUG
        std::cout<<" No vertex found in event "<<std::endl; 
#endif
       return dimuon; 
       } else {
          // cout<<" Event vertex is selected "<<endl;
       }

    dimuon = true;
*/
    return dimuon;

} 
}


