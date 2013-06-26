#ifndef CD_NuclearTrackCorrector_H_
#define CD_NuclearTrackCorrector_H_


// -*- C++ -*-
//
// Package:    NuclearTrackCorrector
// Class:      NuclearTrackCorrector
// 
/**\class NuclearTrackCorrector NuclearTrackCorrector.h RecoTracker/NuclearSeedGenerator/interface/NuclearTrackCorrector.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Loic QUERTENMONT, Vincent ROBERFROID
//         Created:  Tue Sep 18 14:22:48 CEST 2007
// $Id: NuclearTrackCorrector.h,v 1.8 2013/02/27 13:28:32 muzaffar Exp $
//
//


// system include files
#include <memory>
#include <string>
#include <stdio.h>

// user include files

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/VertexReco/interface/NuclearInteraction.h"
#include "DataFormats/VertexReco/interface/NuclearInteractionFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "RecoTracker/NuclearSeedGenerator/interface/TrajectoryToSeedMap.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/NuclearSeedGenerator/interface/TrackCandidateToTrajectoryMap.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"
#include "RecoTracker/NuclearSeedGenerator/interface/NuclearTrackCorrector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoTracker/CkfPattern/interface/TransientInitialStateEstimator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "RecoTracker/TrackProducer/interface/TrackProducerAlgorithm.h"





class TransientInitialStateEstimator;

//
// class decleration
//

class NuclearTrackCorrector :  public edm::EDProducer {

   public:
      typedef edm::RefVector<TrajectorySeedCollection> TrajectorySeedRefVector;
      typedef edm::Ref<TrajectoryCollection> TrajectoryRef;
      typedef edm::Ref<TrackCandidateCollection> TrackCandidateRef;
      typedef TransientTrackingRecHit::ConstRecHitContainer ConstRecHitContainer;
      typedef TrackProducerAlgorithm<reco::Track>::AlgoProductCollection AlgoProductCollection;

   public:

      explicit NuclearTrackCorrector(const edm::ParameterSet&);
      ~NuclearTrackCorrector();

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() ;

      /// check if the trajectory has to be refitted and get the new trajectory
      bool newTrajNeeded(Trajectory& newtrajectory, const TrajectoryRef& trajRef, const reco::NuclearInteraction& ni);

      /// get a new TrackExtra from an AlgoProductCollection
      reco::TrackExtra getNewTrackExtra(const AlgoProductCollection& algoresults);

      /// Get the refitted track from the Trajectory
      bool getTrackFromTrajectory(const Trajectory& newTraj , const TrajectoryRef& initialTrajRef, AlgoProductCollection& algoResults);

      /// Calculate the inital state to be used to buil the track
      TrajectoryStateOnSurface getInitialState(const reco::Track * theT,
                                            TransientTrackingRecHit::RecHitContainer& hits,
                                            const TrackingGeometry * theG,
                                            const MagneticField * theMF);

      void  swap_map(const  edm::Handle< TrajectoryCollection >& trajColl , std::map< reco::TrackRef, edm::Ref<TrajectoryCollection> >& result);
      
      // ----------member data ---------------------------


      std::string str_Input_Trajectory;
      std::string str_Input_NuclearInteraction;
      int    int_Input_Hit_Distance;

      int    verbosity;
      int    KeepOnlyCorrectedTracks;

      std::vector< std::pair<unsigned int, unsigned int> > Indice_Map;

      
      edm::ESHandle<TrackerGeometry> theG;
      edm::ESHandle<MagneticField> theMF;
      edm::ESHandle<TrajectoryFitter> theFitter;
      edm::ESHandle<Propagator> thePropagator;
      edm::ParameterSet conf_;
      TransientInitialStateEstimator*  theInitialState;

      TrackProducerAlgorithm<reco::Track>* theAlgo;
      const TrajTrackAssociationCollection* m_TrajToTrackCollection;
};

#endif
