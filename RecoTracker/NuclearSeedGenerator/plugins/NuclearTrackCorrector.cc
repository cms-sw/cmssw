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
// Original Author:  Loic QUERTENMONT, Vincent ROBERFROID
//         Created:  Tue Sep 18 14:22:48 CEST 2007
// $Id: NuclearTrackCorrector.cc,v 1.14 2013/02/27 13:28:32 muzaffar Exp $
//
//

#include "RecoTracker/NuclearSeedGenerator/interface/NuclearTrackCorrector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoTracker/CkfPattern/interface/TransientInitialStateEstimator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h" 

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;
using namespace reco;


NuclearTrackCorrector::NuclearTrackCorrector(const edm::ParameterSet& iConfig) :
conf_(iConfig),
theInitialState(0)
{
     str_Input_Trajectory           = iConfig.getParameter<std::string>     ("InputTrajectory");
     str_Input_NuclearInteraction   = iConfig.getParameter<std::string>     ("InputNuclearInteraction");
     verbosity			    = iConfig.getParameter<int>             ("Verbosity");
     KeepOnlyCorrectedTracks        = iConfig.getParameter<bool>             ("KeepOnlyCorrectedTracks");


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
  iSetup.get<TrajectoryFitter::Record>().get(fitterName,theFitter);

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

  edm::Handle< NuclearInteractionCollection > temp_m_NuclearInteractionCollection;
  iEvent.getByLabel( str_Input_NuclearInteraction.c_str(), temp_m_NuclearInteractionCollection );
  const NuclearInteractionCollection m_NuclearInteractionCollection = *(temp_m_NuclearInteractionCollection.product());

  edm::Handle< TrajTrackAssociationCollection > h_TrajToTrackCollection;
  iEvent.getByLabel( str_Input_Trajectory.c_str(), h_TrajToTrackCollection );
  m_TrajToTrackCollection = h_TrajToTrackCollection.product();


  // Correct the trajectories (Remove trajectory's hits that are located after the nuclear interacion)
  // --------------------------------------------------------------------------------------------------
  if(verbosity>=1){
    LogDebug("NuclearTrackCorrector")
      <<"Number of trajectories                    = "<<m_TrajectoryCollection.size() <<std::endl
      <<"Number of nuclear interactions            = "<<m_NuclearInteractionCollection.size();
  }

  std::map<reco::TrackRef,TrajectoryRef> m_TrackToTrajMap; 
  swap_map(temp_m_TrajectoryCollection, m_TrackToTrajMap);

  for(unsigned int i = 0 ; i < m_NuclearInteractionCollection.size() ; i++)
  {
        reco::NuclearInteraction ni =  m_NuclearInteractionCollection[i];
        if( ni.likelihood()<0.4) continue;

        reco::TrackRef primTrackRef = ni.primaryTrack().castTo<reco::TrackRef>();

	TrajectoryRef  trajRef = m_TrackToTrajMap[primTrackRef];

        Trajectory newTraj;
        if( newTrajNeeded(newTraj, trajRef, ni) ) {

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
        else {
           if(!KeepOnlyCorrectedTracks) {
                Output_track->push_back(*primTrackRef);
      		Output_trackextra->push_back( *primTrackRef->extra() );
                Output_traj->push_back(*trajRef);
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

// ------------ method called once each job just after ending the event loop  ------------
void 
NuclearTrackCorrector::endJob() {
}
//----------------------------------------------------------------------------------------
bool NuclearTrackCorrector::newTrajNeeded(Trajectory& newtrajectory, const TrajectoryRef& trajRef, const NuclearInteraction& ni) {

        bool needNewTraj=false;
        reco::Vertex::Point vtx_pos = ni.vertex().position();
        double vtx_pos_mag = sqrt (vtx_pos.X()*vtx_pos.X()+vtx_pos.Y()*vtx_pos.Y()+vtx_pos.Z()*vtx_pos.Z());
        if(verbosity>=2) printf("Nuclear Interaction pos = %f\n",vtx_pos_mag );


        newtrajectory = Trajectory(trajRef->seed(), alongMomentum);

        // Look all the Hits of the trajectory and keep only Hits before seeds
        Trajectory::DataContainer Measurements = trajRef->measurements();
        if(verbosity>=2)LogDebug("NuclearTrackCorrector")<<"Size of Measurements  = "<<Measurements.size();

        for(unsigned int m=Measurements.size()-1 ;m!=(unsigned int)-1 ; m--){

                if(!Measurements[m].recHit()->isValid() )continue;
                GlobalPoint hit_pos = theG->idToDet(Measurements[m].recHit()->geographicalId())->surface().toGlobal(Measurements[m].recHit()->localPosition());

                if(verbosity>=2)printf("Hit pos = %f",hit_pos.mag() );

                if(hit_pos.mag()>vtx_pos_mag){
                         if(verbosity>=2)printf(" X ");
                         needNewTraj=true;
                }else{
                        newtrajectory.push(Measurements[m]);
                }
                if(verbosity>=2)printf("\n");
        }

	return needNewTraj;
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
  
  //avoiding to use transientTrack, it should be faster;
  TrajectoryStateOnSurface innerStateFromTrack=trajectoryStateTransform::innerStateOnSurface(*theT,*theG,theMF);
  TrajectoryStateOnSurface outerStateFromTrack=trajectoryStateTransform::outerStateOnSurface(*theT,*theG,theMF);
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
//----------------------------------------------------------------------------------------
void  NuclearTrackCorrector::swap_map(const  edm::Handle< TrajectoryCollection >& trajColl , std::map< reco::TrackRef, edm::Ref<TrajectoryCollection> >& result) {
  for(unsigned int i = 0 ; i < trajColl->size() ; i++)
  {
     TrajectoryRef      InTrajRef    ( trajColl, i);
     reco::TrackRef  PrimaryTrackRef     = m_TrajToTrackCollection->operator[]( InTrajRef );
     result[ PrimaryTrackRef ] = InTrajRef;
  }
}
