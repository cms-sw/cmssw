#ifndef RecoTracker_FinalTrackSelectors_TrackCollectionCloner_H
#define RecoTracker_FinalTrackSelectors_TrackCollectionCloner_H
/*
 *
 * selects a subset of a track collection, copying extra information on demand
 * to be used moslty as an helper in the produce method of selectors
 *
 * \author Giovanni Petrucciani
 *
 *
 *
 */

#include <utility>
#include <vector>
#include <memory>
#include <algorithm>
#include <map>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"



class TrackCollectionCloner {
public:

  /// copy only the tracks, not extras and rechits (for AOD)
  bool copyExtras_;
  /// copy also trajectories and trajectory->track associations
  bool copyTrajectories_;
 

  struct Tokens {
    Tokens(){}
    Tokens(edm::InputTag const & tag, edm::ConsumesCollector && iC) :
      hSrcTrackToken_( iC.consumes<reco::TrackCollection>( tag ) ),
      hTrajToken_( iC.mayConsume< std::vector<Trajectory> >( tag ) ),
      hTTAssToken_( iC.mayConsume< TrajTrackAssociationCollection >( tag ) ){}
    
    /// source collection label
    edm::EDGetTokenT<reco::TrackCollection> hSrcTrackToken_;
    edm::EDGetTokenT< std::vector<Trajectory> > hTrajToken_;
    edm::EDGetTokenT< TrajTrackAssociationCollection > hTTAssToken_;

  };

  template<typename Producer>
  TrackCollectionCloner(Producer & producer, const edm::ParameterSet & cfg, bool copyDefault ) :
    copyExtras_(cfg.template getUntrackedParameter<bool>("copyExtras", copyDefault)),
    copyTrajectories_(cfg.template getUntrackedParameter<bool>("copyTrajectories", copyDefault)) {
    std::string alias( cfg.getParameter<std::string>( "@module_label" ) );
    producer.template produces<reco::TrackCollection>().setBranchAlias( alias + "Tracks" );
    if (copyExtras_) {
      producer.template produces<reco::TrackExtraCollection>().setBranchAlias( alias + "TrackExtras" );
      producer.template produces<TrackingRecHitCollection>().setBranchAlias( alias + "RecHits" );
    }
    if (copyTrajectories_) {
      producer.template produces< std::vector<Trajectory> >().setBranchAlias( alias + "Trajectories" );
      producer.template produces< TrajTrackAssociationCollection >().setBranchAlias( alias + "TrajectoryTrackAssociations" );
    }
  }

  struct Producer {
    Producer(edm::Event& ievt, TrackCollectionCloner const & cloner);

    ~Producer();

    void operator()(Tokens const & tokens, std::vector<unsigned int> const & selected);

    /// copy only the tracks, not extras and rechits (for AOD)
    bool copyExtras_;
    /// copy also trajectories and trajectory->track associations
    bool copyTrajectories_;
    
    /// the event
    edm::Event& evt;
    // some space
    std::unique_ptr<reco::TrackCollection> selTracks_;
    std::unique_ptr<reco::TrackExtraCollection> selTrackExtras_;
    std::unique_ptr<TrackingRecHitCollection> selHits_;
    std::unique_ptr< std::vector<Trajectory> > selTrajs_;
    std::unique_ptr< TrajTrackAssociationCollection > selTTAss_;
  };


};

#endif
