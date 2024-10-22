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
#include <memory>
#include <algorithm>
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "TrackingTools/PatternTools/interface/TrackCollectionTokens.h"

class TrackCollectionCloner {
public:
  /// copy only the tracks, not extras and rechits (for AOD)
  bool copyExtras_;
  /// copy also trajectories and trajectory->track associations
  bool copyTrajectories_;

  using Tokens = TrackCollectionTokens;

  TrackCollectionCloner(edm::ProducesCollector producesCollector, const edm::ParameterSet& cfg, bool copyDefault)
      : copyExtras_(cfg.template getUntrackedParameter<bool>("copyExtras", copyDefault)),
        copyTrajectories_(cfg.template getUntrackedParameter<bool>("copyTrajectories", copyDefault)) {
    std::string alias(cfg.getParameter<std::string>("@module_label"));
    producesCollector.produces<reco::TrackCollection>().setBranchAlias(alias + "Tracks");
    if (copyExtras_) {
      producesCollector.produces<reco::TrackExtraCollection>().setBranchAlias(alias + "TrackExtras");
      producesCollector.produces<TrackingRecHitCollection>().setBranchAlias(alias + "RecHits");
    }
    if (copyTrajectories_) {
      producesCollector.produces<std::vector<Trajectory> >().setBranchAlias(alias + "Trajectories");
      producesCollector.produces<TrajTrackAssociationCollection>().setBranchAlias(alias +
                                                                                  "TrajectoryTrackAssociations");
    }
  }

  static void fill(edm::ParameterSetDescription& desc);

  struct Producer {
    Producer(edm::Event& ievt, TrackCollectionCloner const& cloner);

    ~Producer();

    void operator()(Tokens const& tokens, std::vector<unsigned int> const& selected);

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
    std::unique_ptr<std::vector<Trajectory> > selTrajs_;
    std::unique_ptr<TrajTrackAssociationCollection> selTTAss_;
  };
};

#endif
