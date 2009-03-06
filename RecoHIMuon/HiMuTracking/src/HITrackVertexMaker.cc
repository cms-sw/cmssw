// -*- C++ -*-
//
// Package:    TestMuL1L2
// Class:      TestMuL1L2
//
// \class TestMuL1L2 TestMuL1L2.cc TestMuonL1L2/TestMuL1L2/src/TestMuL1L2.cc
//
// Original Author:  Dong Ho Moon
//         Created:  Wed May  9 06:22:36 CEST 2007
// $Id: HITrackVertexMaker.cc,v 1.4 2008/07/04 08:26:26 kodolova Exp $
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
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateClosestToBeamLineBuilder.h"
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

   candTag_          = ps1.getParameter< edm::InputTag > ("CandTag");
   rphirecHitsTag    = ps1.getParameter<edm::InputTag>("rphiRecHits");
#ifdef DEBUG
   std::cout<<" Start HI TrackVertexMaker constructor "<<std::endl;
#endif
   pset_ = ps1;

//
// Initializetion from Records
    es1.get<TrackerRecoGeometryRecord>().get( tracker );
    es1.get<IdealMagneticFieldRecord>().get(magfield);
    es1.get<CkfComponentsRecord>().get("",measurementTrackerHandle);
    es1.get<TransientRecHitRecord>().get("WithoutRefit",recHitBuilderHandle); 
       
    double theChiSquareCut = 500.;
    double nsig = 3.;
    int theLowMult = 1;
    theEstimator = new HICMeasurementEstimator(theChiSquareCut, nsig, &(*tracker), &(*magfield));
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
#ifdef DEBUG
    std::cout<<" HICTrajectoryBuilder constructed "<<std::endl;
#endif
}



//Destructor

HITrackVertexMaker::~HITrackVertexMaker()
{
//   std::cout<<" Destructor starts "<<std::endl;
//   delete theTrajectoryBuilder;
//   delete theMinPtFilter;
//   delete theEstimator;
//   std::cout<<" Destructor ends "<<std::endl; 
} 

bool HITrackVertexMaker::produceTracks(const edm::Event& e1, const edm::EventSetup& es1, HICConst* theHICConst, FmpConst* theFmpConst)
{
    
   bool dimuon = false;

   cout << " Vertex is set to "<<theHICConst->zvert<<endl;
 
//
// Get recHit builder
//

  es1.get<TransientRecHitRecord>().get("WithoutRefit",recHitBuilderHandle);

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
//
// Get L1 muon info
//
  vector<L1MuGMTExtendedCand> excall;


/*  
try{  
edm::Handle<L1MuGMTReadoutCollection> gmtrc1_handle;
e1.getByLabel("gmtDigis",gmtrc1_handle);

L1MuGMTReadoutCollection const* gmtrc1 = gmtrc1_handle.product();


vector<L1MuGMTReadoutRecord> gmtrecords = gmtrc1->getRecords();
vector<L1MuGMTReadoutRecord>::const_iterator igmtrr1;
std::cout<<" Number of :L1 "<<gmtrecords.size()<<std::endl;


for(igmtrr1=gmtrecords.begin(); igmtrr1!=gmtrecords.end(); igmtrr1++)
{

vector<L1MuRegionalCand>::const_iterator iter11;
vector<L1MuRegionalCand> rmc1;

vector<L1MuGMTExtendedCand>::const_iterator gmt_iter1; 
vector<L1MuGMTExtendedCand> exc1 = igmtrr1->getGMTCands();

std::cout<<" Number of :L1MuGMTExtendedCand "<<exc1.size()<<std::endl;
int igmt1 = 0;
for(gmt_iter1 = exc1.begin(); gmt_iter1!=exc1.end(); gmt_iter1++)
	{
	double etal1=(*gmt_iter1).etaValue();
	double phil1=(*gmt_iter1).phiValue();
	int chargel1=(*gmt_iter1).charge();
	double ptl1=(*gmt_iter1).ptValue();
	cout<<"etal1= "<<etal1<<"phil1= "<<phil1<<"ptl1= "<<ptl1<<"chargel1= "<<chargel1<<endl;
	igmt1++;
        excall.push_back(*gmt_iter1);

	} 
}

} 
        catch (cms::Exception& e) { // can't find it!
              throw e;
       }


   edm::Handle<RecoChargedCandidateCollection> mucands;
   e1.getByLabel (candTag_,mucands);
   RecoChargedCandidateCollection::const_iterator cand1;
   RecoChargedCandidateCollection::const_iterator cand2;
*/   
   
   
   edm::Handle<reco::TrackCollection> mucands;
   e1.getByLabel (candTag_,mucands);
   reco::TrackCollection::const_iterator cand1;
   reco::TrackCollection::const_iterator cand2;
   
      
   cout<<" Number of muon candidates "<<mucands->size()<<" "<<excall.size()<<endl;

//   if(mucands->size() == 0 && excall.size() == 0) return dimuon;

   if( mucands->size() < 2 ) return dimuon;
   
// For trajectory builder   
   int  theLowMult = 1;
   theEstimator->setHICConst(theHICConst);
   theEstimator->setMult(theLowMult);
   theNavigationSchool = new HICSimpleNavigationSchool(&(*tracker), &(*magfield));
   theTrajectoryBuilder->settracker(measurementTrackerHandle.product());
//============================   
    FastMuPropagator* theFmp = new FastMuPropagator(&(*magfield),theFmpConst); 
    StateOnTrackerBound state(theFmp); 
    TrajectoryStateOnSurface tsos;
    
#ifdef DEBUG   
//   for (cand1=mucands->begin(); cand1!=mucands->end(); cand1++) {
//     reco::TrackRef mytr = (*cand1).track(); 
//      std::cout<<" Inner position "<<(*mytr).innerPosition().x()<<" "<<(*mytr).innerPosition().y()<<" "<<(*mytr).innerPosition().z()<<std::endl;
//
//   } 
#endif 
  
    HICFTSfromL1orL2 vFts(&(*magfield));
    
    
    int NumOfSigma=4;
    HICTkOuterStartingLayerFinder TkOSLF(NumOfSigma, &(*magfield), &(*tracker), theHICConst);
   
    int mult = 1;
    DiMuonSeedGeneratorHIC Seed(rphirecHitsTag,&(*magfield),&(*tracker), theHICConst, mult);

    vector<FreeTrajectoryState> theFts = vFts.createFTSfromStandAlone((*mucands));

#ifdef DEBUG
    cout<<" Size of the freeTS "<<theFts.size()<<endl;
#endif 
   
   
    
   DiMuonSeedGeneratorHIC::SeedContainer myseeds;  
   map<DetLayer*, DiMuonSeedGeneratorHIC::SeedContainer> seedmap;
   vector<Trajectory> allTraj;
   
   int numofseeds = 0; 
   for(vector<FreeTrajectoryState>::iterator ifts=theFts.begin(); ifts!=theFts.end(); ifts++)
   {
#ifdef DEBUG
     cout<<" cycle on Muon Trajectory State " <<(*ifts).parameters().position().perp()<<
                                          " " <<(*ifts).parameters().position().z()   <<endl;
#endif
     tsos=state((*ifts));
     
     int iseedn = 0; 
     
     if(tsos.isValid())
     {
     
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

	if( seedlayers.size() == 0 ) continue;
	
//	DiMuonSeedGeneratorHIC::SeedContainer seeds = Seed.produce(e1 ,es1, (*ftsnew), tsos, (*ifts), 
//	                                      recHitBuilderHandle.product(), measurementTrackerHandle.product(), &seedlayers);

        map<DetLayer*, DiMuonSeedGeneratorHIC::SeedContainer> seedmap = Seed.produce(e1 ,es1, (*ftsnew), tsos, (*ifts), 
	                                       recHitBuilderHandle.product(), measurementTrackerHandle.product(), &seedlayers);
	
	int ilnum = 0;
	  for( vector<DetLayer*>::iterator it = seedlayers.begin(); it != seedlayers.end(); it++)
	  {
	    ilnum++;
	    vector<Trajectory> theTmpTrajectories0;
	    DiMuonSeedGeneratorHIC::SeedContainer seeds = (*seedmap.find(*it)).second;
    // set the correct navigation
            NavigationSetter setter( *theNavigationSchool);
#ifdef DEBUG
       std::cout<<" Layer "<<ilnum<<" Number of seeds in layer "<<seeds.size()<<std::endl;	    
#endif	    
	    if(seeds.size() > 0) {
       for(DiMuonSeedGeneratorHIC::SeedContainer::iterator iseed=seeds.begin();iseed!=seeds.end();iseed++)
       {
         std::vector<TrajectoryMeasurement> theV = (*iseed).measurements();
#ifdef DEBUG
         std::cout<< "Layer " <<ilnum<<" Seed number "<<iseedn<<"position r "<<theV[0].recHit()->globalPosition().perp()<<" phi "<<theV[0].recHit()->globalPosition().phi()<<" z "<<
         theV[0].recHit()->globalPosition().z()<<" momentum "<<theV[0].updatedState().freeTrajectoryState()->parameters().momentum().perp()<<" "<<
         theV[0].updatedState().freeTrajectoryState()->parameters().momentum().z()<<std::endl;
#endif
	 vector<Trajectory> theTmpTrajectories = theTrajectoryBuilder->trajectories(*iseed);   
#ifdef DEBUG
        cout<<" Number of found trajectories "<<theTmpTrajectories.size()<<endl;	 
#endif
        if(theTmpTrajectories.size()>0) {
	                                    allTraj.insert(allTraj.end(),theTmpTrajectories.begin(),theTmpTrajectories.end());
					    theTmpTrajectories0.insert(theTmpTrajectories0.end(),theTmpTrajectories.begin(),theTmpTrajectories.end());
	                                }
	if(theTmpTrajectories0.size() > 0) { std::cout<<"We found trajectories for at least one seed "<<std::endl; break;}
					
        iseedn++;
	    
       } // seeds 	    
	    } // there are seeds/layer
	    
//	    std::cout<<" Size of seed container "<<seeds.size()<<" size of trajectories "<<theTmpTrajectories0.size()<<std::endl;  
	    
	    if( theTmpTrajectories0.size() > 0 ) break;
	    
	  } // Detlayer cycle  
	
     } // tsos. isvalid
     if(iseedn > 0) numofseeds++;
   } // Muon Free trajectory state

   cout<<" Number of muons trajectories in MUON "<<theFts.size()<<" Number of seeds "<<numofseeds+1<<endl;
        
//
// start fitting procedure
//
#ifdef DEBUG
    if(allTraj.size()>0 ) {std::cout<<" Event reconstruction finished with "<<allTraj.size()<<std::endl;} else {std::cout<<" Event reconstruction::no tracks found"<<std::endl;}
#endif
    if(allTraj.size()<2)  return dimuon;

    edm::ESHandle<GlobalTrackingGeometry> globTkGeomHandle;
    es1.get<GlobalTrackingGeometryRecord>().get(globTkGeomHandle);

    vector<reco::Track> thePositiveTracks;
    vector<reco::Track> theNegativeTracks; 
    vector<reco::TrackRef> thePositiveTrackRefs;
    vector<reco::TrackRef> theNegativeTrackRefs;
    vector<reco::TransientTrack> thePositiveTransTracks;
    vector<reco::TransientTrack> theNegativeTransTracks;

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

    edm::ESHandle<TrajectoryFitter> theFitterTrack;
    edm::ESHandle<TrajectorySmoother> theSmootherTrack;
    edm::ESHandle<Propagator> thePropagatorTrack;
    es1.get<TrackingComponentsRecord>().get("KFFitterForRefitInsideOut",theFitterTrack);
    es1.get<TrackingComponentsRecord>().get("KFSmootherForRefitInsideOut",theSmootherTrack);
    es1.get<TrackingComponentsRecord>().get("SmartPropagatorAny",thePropagatorTrack);
    enum RefitDirection{insideOut,outsideIn,undetermined};


    for(vector<Trajectory>::iterator it = allTraj.begin(); it!= allTraj.end(); it++)
    {
//
// Refit the trajectory
//
// Refit the trajectory before putting into the last result

  Trajectory::ConstRecHitContainer recHitsForReFit = (*it).recHits();
  
  PTrajectoryStateOnDet garbage1;
  edm::OwnVector<TrackingRecHit> garbage2;
  TrajectoryStateOnSurface firstTSOS = (*it).firstMeasurement().updatedState();

//   cout<<" firstTSOS "<<firstTSOS.freeTrajectoryState()->position().perp()<<" "<<firstTSOS.freeTrajectoryState()->position().z()<<endl;
//  PropagationDirection propDir =
//    (firstTSOS.globalPosition().basicVector().dot(firstTSOS.globalMomentum().basicVector())>0) ? alongMomentum : oppositeToMomentum;

  PropagationDirection propDir = oppositeToMomentum;
  
//  RefitDirection theRefitDirection = insideOut;
//  propDir = alongMomentum;
//  RefitDirection theRefitDirection = outsideIn;

//  if(propDir == alongMomentum && theRefitDirection == outsideIn)  {cout<<" oppositeToMomentum "<<endl; propDir=oppositeToMomentum;}
//  if(propDir == oppositeToMomentum && theRefitDirection == insideOut) {cout<<" alongMomentum "<<endl; propDir=alongMomentum;}

  TrajectorySeed seed(garbage1,garbage2,propDir);
  vector<Trajectory> trajectories = theFitterTrack->fit(seed,recHitsForReFit,firstTSOS);

//  vector<Trajectory> trajectories = theFitterTrack->fit(*it);

//  if(trajectories.empty()) {cout<<" No refit is done "<<endl;};
  
//  Trajectory trajectoryBW = trajectories.front();
//  vector<Trajectory> trajectoriesSM = theSmoother->trajectories(trajectoryBW);
//  if(trajectoriesSM.size()<1) continue;

    TrajectoryStateOnSurface innertsos;
    if (it->direction() == alongMomentum) {
   //    cout<<" Direction is along momentum "<<endl;
      innertsos = it->firstMeasurement().updatedState();
    } else {
      innertsos = it->lastMeasurement().updatedState();
   //   cout<<" Direction is opposite to momentum "<<endl;
      
    }
   //  cout<<" Position of the innermost point "<<innertsos.freeTrajectoryState()->position().perp()<<" "<<innertsos.freeTrajectoryState()->position().z()<<endl;
    TrajectoryStateClosestToBeamLineBuilder tscblBuilder;
    TrajectoryStateClosestToBeamLine tscbl = tscblBuilder(*(innertsos.freeState()),bs);

    if (tscbl.isValid()==false) {
    //cout<<" false track "<<endl; 
    continue;}

    GlobalPoint v = tscbl.trackStateAtPCA().position();
    math::XYZPoint  pos( v.x(), v.y(), v.z() );
    
//    cout<<" Position of track close to vertex "<<v.perp()<<" "<<v.z()<<" Primary vertex "<<theHICConst->zvert<<
//                                     " charge "<<tscbl.trackStateAtPCA().charge()<<endl;
   
    if(v.perp() > 0.1 ) continue;
    if(fabs(v.z() - theHICConst->zvert ) > 0.4 ) continue;

    GlobalVector p = tscbl.trackStateAtPCA().momentum();
    math::XYZVector mom( p.x(), p.y(), p.z() );

    Track theTrack(it->chiSquared(),
                   it->ndof(), 
                   pos, mom, tscbl.trackStateAtPCA().charge(), tscbl.trackStateAtPCA().curvilinearError(),Algo);
    TransientTrack tmpTk( theTrack, &(*magfield), globTkGeomHandle );
    if( tscbl.trackStateAtPCA().charge() > 0)
    {
      thePositiveTracks.push_back( theTrack );
      thePositiveTransTracks.push_back( tmpTk );
    }
      else
      {
        theNegativeTracks.push_back( theTrack );
        theNegativeTransTracks.push_back( tmpTk );
      }
    } // Traj

   if( thePositiveTracks.size() < 1 || theNegativeTracks.size() < 1 ){ cout<<" No enough tracks to get vertex "<<endl; return dimuon; }

   bool useRefTrax=true;
   KalmanVertexFitter theFitter(useRefTrax);
   TransientVertex theRecoVertex;
//
// Create possible two particles vertices
//
   vector<reco::TransientTrack> theTwoTransTracks;
   vector<TransientVertex> theVertexContainer;
   for(vector<reco::TransientTrack>::iterator iplus = thePositiveTransTracks.begin(); iplus != thePositiveTransTracks.end(); iplus++)
   {
     theTwoTransTracks.clear();
     theTwoTransTracks.push_back(*iplus);
     for(vector<reco::TransientTrack>::iterator iminus = theNegativeTransTracks.begin(); iminus != theNegativeTransTracks.end(); iminus++)
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
      theVertexContainer.push_back(theRecoVertex);
     } // iminus
  } // iplus
  
       if( theVertexContainer.size() < 1 ) { 
        std::cout<<" No vertex found in event "<<std::endl; 
       return dimuon; 
       } else {
          // cout<<" Event vertex is selected "<<endl;
       }


    dimuon = true;
    return dimuon;
} 
}


