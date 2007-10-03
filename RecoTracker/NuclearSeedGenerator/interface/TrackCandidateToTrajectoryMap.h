#ifndef TrackCandidateToTrajectoryMap_h
#define TrackCandidateToTrajectoryMap_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/AssociationMap.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "RecoTracker/NuclearSeedGenerator/interface/TrajectoryToSeedMap.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"



// TrajectoryToTrajectoryMap

/// association map
typedef edm::AssociationMap<edm::OneToOne<TrajectoryCollection, TrajectoryCollection> > TrajectoryToTrajectoryMap;
typedef  TrajectoryToTrajectoryMap::value_type TrajectoryToTrajectory;

/// reference to an object in a collection of TrajectoryMap objects
typedef edm::Ref<TrajectoryToTrajectoryMap> TrajectoryToTrajectoryMapRef;

/// reference to a collection of TrajectoryMap object
typedef edm::RefProd<TrajectoryToTrajectoryMap> TrajectoryToTrajectoryMapRefProd;

/// vector of references to objects in the same colletion of SeedMap objects
typedef edm::RefVector<TrajectoryToTrajectoryMap> TrajectoryToTrajectoryMapRefVector;



// TrackCandidateToTrajectoryMap

/// association map
typedef edm::AssociationMap<edm::OneToOne<TrackCandidateCollection, TrajectoryCollection> > TrackCandidateToTrajectoryMap;
typedef  TrackCandidateToTrajectoryMap::value_type TrackCandidateToTrajectory;

/// reference to an object in a collection of TrajectoryMap objects
typedef edm::Ref<TrackCandidateToTrajectoryMap> TrackCandidateToTrajectoryMapRef;

/// reference to a collection of TrajectoryMap object
typedef edm::RefProd<TrackCandidateToTrajectoryMap> TrackCandidateToTrajectoryMapRefProd;

/// vector of references to objects in the same colletion of SeedMap objects
typedef edm::RefVector<TrackCandidateToTrajectoryMap> TrackCandidateToTrajectoryMapRefVector;



// TrackToTrajectoryMap

/// association map
typedef edm::AssociationMap<edm::OneToOne<reco::TrackCollection, TrajectoryCollection> > TrackToTrajectoryMap;
typedef  TrackToTrajectoryMap::value_type TrackToTrajectory;

/// reference to an object in a collection of TrajectoryMap objects
typedef edm::Ref<TrackToTrajectoryMap> TrackToTrajectoryMapRef;

/// reference to a collection of TrajectoryMap object
typedef edm::RefProd<TrackToTrajectoryMap> TrackToTrajectoryMapRefProd;

/// vector of references to objects in the same colletion of SeedMap objects
typedef edm::RefVector<TrackToTrajectoryMap> TrackToTrajectoryMapRefVector;


/// association map
   typedef edm::AssociationMap<edm::OneToOne<reco::TrackCollection, reco::TrackCollection> > TrackToTrackMap;
   typedef  TrackToTrackMap::value_type TrackToTrack;

   /// reference to an object in a collection of SeedMap objects
  typedef edm::Ref<TrackToTrackMap> TrackToTrackMapRef;

  /// reference to a collection of SeedMap object
  typedef edm::RefProd<TrackToTrackMap> TrackToTrackMapRefProd;

  /// vector of references to objects in the same colletion of SeedMap objects
  typedef edm::RefVector<TrackToTrackMap> TrackToTrackMapRefVector;



  
#endif


