// -*- C++ -*-
//
// Package:    NuclearTrackCorrector
// Class:      NuclearTrackCorrector
// 
/**\class NuclearTrackCorrector NuclearTrackCorrector.cc RecoTracker/NuclearSeedGenerator/plugin/NuclearTrackCorrector.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Loic QUERTENMONT
//         Created:  Tue Sep 18 14:22:48 CEST 2007
// $Id: NuclearTrackCorrector.cc,v 1.6 2007/10/08 15:52:14 roberfro Exp $
//
//

#include "RecoTracker/NuclearSeedGenerator/interface/NuclearTrackCorrector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoTracker/CkfPattern/interface/TransientInitialStateEstimator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h" 

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

using namespace edm;
using namespace std;
using namespace reco;


NuclearTrackCorrector::NuclearTrackCorrector(const edm::ParameterSet& iConfig) :
conf_(iConfig),
theInitialState(0)
{
     str_Input_Trajectory           = iConfig.getParameter<std::string>     ("InputTrajectory");
     str_Input_NuclearSeed          = iConfig.getParameter<std::string>     ("InputNuclearSeed");
     int_Input_Hit_Distance         = iConfig.getParameter<int> 	    ("InputHitDistance");
     verbosity			    = iConfig.getParameter<int>             ("Verbosity");
     KeepOnlyCorrectedTracks        = iConfig.getParameter<int>             ("KeepOnlyCorrectedTracks");


     theAlgo = new TrackProducerAlgorithm<reco::Track>(iConfig);

     produces< TrajectoryCollection >();
     produces< TrajectoryToTrajectoryMap >();

     produces< reco::TrackExtraCollection >();
     produces< reco::TrackCollection >();       
     produces< TrackToTrajectoryMap >();

     produces< TrackToTrackMap >();
}


NuclearTrackCorrector::~NuclearTrackCorrector()
{
}

void
NuclearTrackCorrector::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // Create Output Collections
  // --------------------------------------------------------------------------------------------------
  std::auto_ptr<TrajectoryCollection>           Output_traj          ( new TrajectoryCollection );
  std::auto_ptr<TrajectoryToTrajectoryMap>      Output_trajmap       ( new TrajectoryToTrajectoryMap );

  std::auto_ptr<reco::TrackExtraCollection>	Output_trackextra    ( new reco::TrackExtraCollection );
  std::auto_ptr<reco::TrackCollection>          Output_track         ( new reco::TrackCollection );
  std::auto_ptr<TrackToTrajectoryMap>           Output_trackmap      ( new TrackToTrajectoryMap );

  std::auto_ptr<TrackToTrackMap>  	        Output_tracktrackmap ( new TrackToTrackMap );





  // Load Reccord
  // --------------------------------------------------------------------------------------------------
  std::string fitterName = conf_.getParameter<std::string>("Fitter");   
  iSetup.get<TrackingComponentsRecord>().get(fitterName,theFitter);

  std::string propagatorName = conf_.getParameter<std::string>("Propagator");   
  iSetup.get<TrackingComponentsRecord>().get(propagatorName,thePropagator);

  iSetup.get<TrackerDigiGeometryRecord>().get(theG);

  reco::TrackExtraRefProd rTrackExtras = iEvent.getRefBeforePut<reco::TrackExtraCollection>();

   iSetup.get<IdealMagneticFieldRecord>().get(theMF); 

  // Load Inputs
  // --------------------------------------------------------------------------------------------------
  edm::Handle< TrajectoryCollection > temp_m_TrajectoryCollection;
  iEvent.getByLabel( str_Input_Trajectory.c_str(), temp_m_TrajectoryCollection );
  const TrajectoryCollection m_TrajectoryCollection = *(temp_m_TrajectoryCollection.product());

  edm::Handle< TrajectorySeedCollection > temp_m_NuclearSeedCollection;
  iEvent.getByLabel( str_Input_NuclearSeed.c_str(), temp_m_NuclearSeedCollection );
  const TrajectorySeedCollection m_NuclearSeedCollection = *(temp_m_NuclearSeedCollection.product());

  edm::Handle< TrajectoryToSeedsMap > temp_m_TrajectoryToSeedsMap;
  iEvent.getByLabel( str_Input_NuclearSeed.c_str(), temp_m_TrajectoryToSeedsMap );
  const TrajectoryToSeedsMap m_TrajectoryToSeedsMap = *(temp_m_TrajectoryToSeedsMap.product());

  edm::Handle< TrajTrackAssociationCollection > h_TrajToTrackCollection;
  iEvent.getByLabel( str_Input_Trajectory.c_str(), h_TrajToTrackCollection );
  m_TrajToTrackCollection = h_TrajToTrackCollection.product();


  // Correct the trajectories (Remove trajectory's hits that are located after the nuclear interacion)
  // --------------------------------------------------------------------------------------------------
  if(verbosity>=1){
  printf("Number of trajectories                    = %i\n",m_TrajectoryCollection.size() );
  printf("Number of seeds                           = %i\n",m_NuclearSeedCollection.size() );
  printf("Number of tracjectories attached to seeds = %i\n",m_TrajectoryToSeedsMap.size() );
  }

  for(unsigned int i = 0 ; i < m_TrajectoryCollection.size() ; i++)
  {

	TrajectoryRef  trajRef( temp_m_TrajectoryCollection, i );

        TrajectorySeedRefVector seedRef;
	try{
	       	seedRef = m_TrajectoryToSeedsMap [ trajRef ];
	}
	catch(edm::Exception event){}
 
        Trajectory newTraj;
        if( newTrajNeeded(newTraj, trajRef, seedRef) ) {

          AlgoProductCollection   algoResults; 
          bool isOK = getTrackFromTrajectory( newTraj , trajRef, algoResults);

          if( isOK ) {

		pair<unsigned int, unsigned int> tempory_pair;
		tempory_pair.first  = Output_track->size();
		tempory_pair.second = i;
		Indice_Map.push_back(tempory_pair);

                reco::TrackExtraRef teref= reco::TrackExtraRef ( rTrackExtras, i );
                reco::TrackExtra newTrackExtra = getNewTrackExtra(algoResults);
                (algoResults[0].second.first)->setExtra( teref ); 

                Output_track->push_back(*algoResults[0].second.first);        
                Output_trackextra->push_back( newTrackExtra );
	        Output_traj->push_back(newTraj);

          }
        }

  }
  const edm::OrphanHandle<TrajectoryCollection>     Handle_traj = iEvent.put(Output_traj);
  const edm::OrphanHandle<reco::TrackCollection> Handle_tracks = iEvent.put(Output_track);
  iEvent.put(Output_trackextra);

  // Make Maps between elements
  // --------------------------------------------------------------------------------------------------
  if(Handle_tracks->size() != Handle_traj->size() )
  {
     printf("ERROR Handle_tracks->size() != Handle_traj->size() \n");
     return;
  }



  for(unsigned int i = 0 ; i < Indice_Map.size() ; i++)
  {
        TrajectoryRef      InTrajRef    ( temp_m_TrajectoryCollection, Indice_Map[i].second );
        TrajectoryRef      OutTrajRef   ( Handle_traj, Indice_Map[i].first );
        reco::TrackRef     TrackRef     ( Handle_tracks, Indice_Map[i].first );

        Output_trajmap ->insert(OutTrajRef,InTrajRef);
        Output_trackmap->insert(TrackRef,InTrajRef);

        try{
                reco::TrackRef  PrimaryTrackRef     = m_TrajToTrackCollection->operator[]( InTrajRef );
	        Output_tracktrackmap->insert(TrackRef,PrimaryTrackRef);
        }catch(edm::Exception event){}
	
  }
  iEvent.put(Output_trajmap);
  iEvent.put(Output_trackmap);
  iEvent.put(Output_tracktrackmap);


  if(verbosity>=3)printf("-----------------------\n");
}

// ------------ method called once each job just before starting event loop  ------------
void 
NuclearTrackCorrector::beginJob(const edm::EventSetup& iSetup)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
NuclearTrackCorrector::endJob() {
}
//----------------------------------------------------------------------------------------
bool NuclearTrackCorrector::newTrajNeeded(Trajectory& newtrajectory, const TrajectoryRef& trajRef, const TrajectorySeedRefVector& seedRef) {

 // Find radius of the inner seed
        double min_seed_radius = 999;
        bool needNewTraj=false;
        for(unsigned int k=0;k<seedRef.size();k++)
        {
                BasicTrajectorySeed::range seed_RecHits = seedRef[k]->recHits();

                if(seedRef[k]->nHits()==0) continue;

                GlobalPoint pos = theG->idToDet(seed_RecHits.first->geographicalId())->surface().toGlobal(seed_RecHits.first->localPosition());
                double seed_radius = sqrt( pos.x()*pos.x() + pos.y()*pos.y() );
                if(seed_radius<min_seed_radius) min_seed_radius = seed_radius;
        }
        if(verbosity>=2) printf("Min Seed Radius = %f\n",min_seed_radius );


        newtrajectory = Trajectory(trajRef->seed(), alongMomentum);

        // Look all the Hits of the trajectory and keep only Hits before seeds
        Trajectory::DataContainer Measurements = trajRef->measurements();
        if(verbosity>=2)printf("Size of Measurements  = %i\n",Measurements.size() );
        for(unsigned int m=Measurements.size()-1 ;m!=(unsigned int)-1 ; m--){

                if(!Measurements[m].recHit()->isValid() )continue;
                GlobalPoint pos = theG->idToDet(Measurements[m].recHit()->geographicalId())->surface().toGlobal(Measurements[m].recHit()->localPosition());

                double hit_radius = sqrt( pow(pos.x(),2) + pow(pos.y(),2) );
                if(verbosity>=2)printf("Hit Radius = %f",hit_radius );
                if(hit_radius>min_seed_radius-int_Input_Hit_Distance){
                         if(verbosity>=2)printf(" X ");
                         needNewTraj=true;
                }else{
                        newtrajectory.push(Measurements[m]);
                }
                if(verbosity>=2)printf("\n");
        }

	if(KeepOnlyCorrectedTracks)  return needNewTraj;

	return 1;
}

//----------------------------------------------------------------------------------------
bool NuclearTrackCorrector::getTrackFromTrajectory(const Trajectory& newTraj , const TrajectoryRef& initialTrajRef, AlgoProductCollection& algoResults) {

        const Trajectory*  it = &newTraj;

        TransientTrackingRecHit::RecHitContainer hits;
        it->validRecHits( hits  );
        

	float ndof=0;
        for(unsigned int h=0 ; h<hits.size() ; h++)
        {
            if( hits[h]->isValid() )
            {
                ndof = ndof + hits[h]->dimension() * hits[h]->weight();
            }
            else {
                 LogDebug("NuclearSeedGenerator") << " HIT IS INVALID ???";
            }
        }


        ndof = ndof - 5;
        reco::TrackRef  theT     = m_TrajToTrackCollection->operator[]( initialTrajRef );
        LogDebug("NuclearSeedGenerator") << " TrackCorrector - number of valid hits" << hits.size() << "\n"
                                         << "                - number of hits from Track " << theT->recHitsSize() << "\n"
                                         << "                - number of valid hits from initial track " << theT->numberOfValidHits();


        if(  hits.size() > 1){

		TrajectoryStateOnSurface theInitialStateForRefitting = getInitialState(&(*theT),hits,theG.product(),theMF.product()
);

           reco::BeamSpot bs;
           return theAlgo->buildTrack(theFitter.product(), thePropagator.product(), algoResults, hits, theInitialStateForRefitting ,it->seed(), ndof, bs, theT->seedRef());
         }

	return false;
}
//----------------------------------------------------------------------------------------
reco::TrackExtra NuclearTrackCorrector::getNewTrackExtra(const AlgoProductCollection& algoResults) {
                Trajectory* theTraj          = algoResults[0].first;
                PropagationDirection seedDir = algoResults[0].second.second;

                TrajectoryStateOnSurface outertsos;
                TrajectoryStateOnSurface innertsos;
                unsigned int innerId, outerId;
                if (theTraj->direction() == alongMomentum) {
                  outertsos = theTraj->lastMeasurement().updatedState();
                  innertsos = theTraj->firstMeasurement().updatedState();
                  outerId   = theTraj->lastMeasurement().recHit()->geographicalId().rawId();
                  innerId   = theTraj->firstMeasurement().recHit()->geographicalId().rawId();
                } else {
                  outertsos = theTraj->firstMeasurement().updatedState();
                  innertsos = theTraj->lastMeasurement().updatedState();
                  outerId   = theTraj->firstMeasurement().recHit()->geographicalId().rawId();
                  innerId   = theTraj->lastMeasurement().recHit()->geographicalId().rawId();
                }

                GlobalPoint v = outertsos.globalParameters().position();
                GlobalVector p = outertsos.globalParameters().momentum();
                math::XYZVector outmom( p.x(), p.y(), p.z() );
                math::XYZPoint  outpos( v.x(), v.y(), v.z() );
                v = innertsos.globalParameters().position();
                p = innertsos.globalParameters().momentum();
                math::XYZVector inmom( p.x(), p.y(), p.z() );
                math::XYZPoint  inpos( v.x(), v.y(), v.z() );

                return reco::TrackExtra (outpos, outmom, true, inpos, inmom, true,
                                        outertsos.curvilinearError(), outerId,
                                        innertsos.curvilinearError(), innerId, seedDir);

}
//----------------------------------------------------------------------------------------
TrajectoryStateOnSurface NuclearTrackCorrector::getInitialState(const reco::Track * theT,
                                                                 TransientTrackingRecHit::RecHitContainer& hits,
                                                                 const TrackingGeometry * theG,
                                                                 const MagneticField * theMF){

  TrajectoryStateOnSurface theInitialStateForRefitting;
  //the starting state is the state closest to the first hit along seedDirection.
  TrajectoryStateTransform transformer;
  //avoiding to use transientTrack, it should be faster;
  TrajectoryStateOnSurface innerStateFromTrack=transformer.innerStateOnSurface(*theT,*theG,theMF);
  TrajectoryStateOnSurface outerStateFromTrack=transformer.outerStateOnSurface(*theT,*theG,theMF);
  TrajectoryStateOnSurface initialStateFromTrack = 
    ( (innerStateFromTrack.globalPosition()-hits.front()->globalPosition()).mag2() <
      (outerStateFromTrack.globalPosition()-hits.front()->globalPosition()).mag2() ) ? 
    innerStateFromTrack: outerStateFromTrack;       
  
  // error is rescaled, but correlation are kept.
  initialStateFromTrack.rescaleError(100);
  theInitialStateForRefitting = TrajectoryStateOnSurface(initialStateFromTrack.localParameters(),
                                                         initialStateFromTrack.localError(),                  
                                                         initialStateFromTrack.surface(),
                                                         theMF); 
  return theInitialStateForRefitting;
}
