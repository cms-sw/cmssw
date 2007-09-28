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
// $Id$
//
//

#include "RecoTracker/NuclearSeedGenerator/interface/NuclearTrackCorrector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoTracker/CkfPattern/interface/TransientInitialStateEstimator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h" 


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

     theAlgo = new TrackProducerAlgorithm(iConfig);



     produces< TrajectoryCollection >();
     produces< TrajectoryToTrajectoryMap >();

     produces< TrackCandidateCollection >();
     produces< TrackCandidateToTrajectoryMap >();

     produces< reco::TrackCollection >();
     produces< TrackToTrajectoryMap >();

     produces< TrackToTracksMap >();
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

  std::auto_ptr<TrackCandidateCollection>       Output_trackCand     ( new TrackCandidateCollection );
  std::auto_ptr<TrackCandidateToTrajectoryMap>  Output_trackCandmap  ( new TrackCandidateToTrajectoryMap );

  std::auto_ptr<reco::TrackCollection>          Output_track         ( new reco::TrackCollection );
  std::auto_ptr<TrackToTrajectoryMap>           Output_trackmap      ( new TrackToTrajectoryMap );

  std::auto_ptr<TrackToTracksMap>  	        Output_tracktrackmap ( new TrackToTracksMap );





  // Load Reccord
  // --------------------------------------------------------------------------------------------------
  edm::ESHandle<TrajectoryFitter> theFitter;
  std::string fitterName = conf_.getParameter<std::string>("Fitter");   
  iSetup.get<TrackingComponentsRecord>().get(fitterName,theFitter);

  edm::ESHandle<Propagator> thePropagator;
  std::string propagatorName = conf_.getParameter<std::string>("Propagator");   
  iSetup.get<TrackingComponentsRecord>().get(propagatorName,thePropagator);

  iSetup.get<TrackerDigiGeometryRecord>().get(pDD);





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
  const TrajTrackAssociationCollection m_TrajToTrackCollection = *(h_TrajToTrackCollection.product());






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

	// Find radius of the inner seed
	double min_seed_radius = 999;
	for(unsigned int k=0;k<seedRef.size();k++)
	{
		BasicTrajectorySeed::range seed_RecHits = seedRef[k]->recHits();
		
		if(seedRef[k]->nHits()==0) continue;
  
		GlobalPoint pos = pDD->idToDet(seed_RecHits.first->geographicalId())->surface().toGlobal(seed_RecHits.first->localPosition());
		double seed_radius = sqrt( pow(pos.x(),2) + pow(pos.y(),2) );
		if(seed_radius<min_seed_radius) min_seed_radius = seed_radius;
	}
        if(verbosity>=2) printf("Min Seed Radius = %f\n",min_seed_radius );


        Trajectory     newtrajectory (trajRef->seed());

        // Look all the Hits of the trajectory and keep only Hits before seeds
	Trajectory::DataContainer Measurements = trajRef->measurements();
	if(verbosity>=2)printf("Size of Measurements  = %i\n",Measurements.size() );
        for(unsigned int m=Measurements.size()-1 ;m!=(unsigned int)-1 ; m--){

		if(!Measurements[m].recHit()->isValid() )continue;
                GlobalPoint pos = pDD->idToDet(Measurements[m].recHit()->geographicalId())->surface().toGlobal(Measurements[m].recHit()->localPosition());
                double hit_radius = sqrt( pow(pos.x(),2) + pow(pos.y(),2) );
                if(verbosity>=2)printf("Hit Radius = %f",hit_radius );
                if(hit_radius>min_seed_radius-int_Input_Hit_Distance){
			 if(verbosity>=2)printf(" X ");
		}else{
			newtrajectory.push(Measurements[m]);
		}
                if(verbosity>=2)printf("\n");
	}

	Output_traj->push_back(newtrajectory);
  }
  const edm::OrphanHandle<TrajectoryCollection>     Handle_traj = iEvent.put(Output_traj);




  //  Convert Trajectory to TrackCandidates
  // --------------------------------------------------------------------------------------------------
  TrajectoryCollection Trajectories1 = *(Handle_traj.product());
  for(unsigned int i = 0 ; i < Trajectories1.size() ; i++)
  {
        Trajectory*  it = &Trajectories1[i];

	Trajectory::RecHitContainer thits;
	it->recHitsV(thits);

	OwnVector<TrackingRecHit> recHits;
	recHits.reserve(thits.size());

	for (Trajectory::RecHitContainer::const_iterator hitIt = thits.begin(); hitIt != thits.end(); hitIt++)
	{
  	    recHits.push_back( (**hitIt).hit()->clone());
	}

	edm::ParameterSet tise_params = conf_.getParameter<edm::ParameterSet>("TransientInitialStateEstimatorParameters") ;
	theInitialState = new TransientInitialStateEstimator( iSetup,  tise_params);
	
	std::pair<TrajectoryStateOnSurface, const GeomDet*> initState = theInitialState->innerState( *it);
      
	// temporary protection againt invalid initial states
	if (! initState.first.isValid() || initState.second == 0)
	{
           cout << "invalid innerState, will not make TrackCandidate" << endl;
           continue;
        }

	PTrajectoryStateOnDet* state = TrajectoryStateTransform().persistentState( initState.first, initState.second->geographicalId().rawId());
	
	Output_trackCand->push_back(TrackCandidate(recHits,it->seed(),*state));
	delete state;
  }       
  const edm::OrphanHandle<TrackCandidateCollection> Handle_trackCand       = iEvent.put(Output_trackCand);




  //  Convert Trajectory to reco::Track
  // --------------------------------------------------------------------------------------------------
  AlgoProductCollection algoResults;
  TrajectoryCollection Trajectories = *(Handle_traj.product());
  for(unsigned int i = 0 ; i < Trajectories.size() ; i++)
  { 
        Trajectory*  it = &Trajectories[i];

      	TransientTrackingRecHit::RecHitContainer hits;
	it->recHitsV( hits  );      
      	float ndof=0;     
	for(unsigned int h=0 ; h<hits.size() ; h++)
	{
	    if( hits[h]->isValid() )
	    {
		ndof = ndof + hits[h]->dimension() * hits[h]->weight();
	    }
      	}      
      	ndof = ndof - 5;


        edm::ParameterSet tise_params = conf_.getParameter<edm::ParameterSet>("TransientInitialStateEstimatorParameters") ;
        theInitialState = new TransientInitialStateEstimator( iSetup,  tise_params);

        std::pair<TrajectoryStateOnSurface, const GeomDet*> initState = theInitialState->innerState( *it);

        // temporary protection againt invalid initial states
        if (! initState.first.isValid() || initState.second == 0)
        {
           cout << "invalid innerState, will not make TrackCandidate" << endl;
           continue;
        }

	bool IsOK = theAlgo->buildTrack(theFitter.product(),thePropagator.product(),algoResults,hits,initState.first,it->seed(),ndof);

	if(IsOK==1){
		Output_track->push_back(*algoResults[0].second.first);
                algoResults.pop_back();
	}else{
	        printf("ERROR during the Trajectory to reco::Track conversion !\n" );
	}
  }
  const edm::OrphanHandle<reco::TrackCollection> Handle_tracks = iEvent.put(Output_track);






  // Make Maps between elements
  // --------------------------------------------------------------------------------------------------
  if(Handle_tracks->size() != Handle_traj->size() || Handle_trackCand->size() != Handle_traj->size() )
  {
     printf("ERROR Handle_tracks->size() != Handle_traj->size() || Handle_trackCand->size() != Handle_traj->size() \n");
     return;
  }

  for(unsigned int i = 0 ; i < Handle_tracks->size() ; i++)
  {
        TrajectoryRef      InTrajRef    ( temp_m_TrajectoryCollection, i );
        TrajectoryRef      OutTrajRef   ( Handle_traj, i );
        reco::TrackRef     TrackRef     ( Handle_tracks, i );
	TrackCandidateRef  TrackCandRef ( Handle_trackCand, i);

        Output_trajmap ->insert(OutTrajRef,InTrajRef);
	Output_trackCandmap->insert(TrackCandRef,InTrajRef);
        Output_trackmap->insert(TrackRef,InTrajRef);

        try{
                reco::TrackRef  PrimaryTrackRef     = m_TrajToTrackCollection[ InTrajRef ];
	        Output_tracktrackmap->insert(TrackRef,PrimaryTrackRef);
        }catch(edm::Exception event){printf("@@@@@@@@@@@   ERROR for getting references @@@@@@@@@@@\n");}
	
  }
  iEvent.put(Output_trajmap);
  iEvent.put(Output_trackCandmap);
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


