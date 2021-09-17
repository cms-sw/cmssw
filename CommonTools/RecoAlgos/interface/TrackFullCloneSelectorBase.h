#ifndef RecoAlgos_TrackFullCloneSelectorBase_h
#define RecoAlgos_TrackFullCloneSelectorBase_h
/** \class TrackFullCloneSelectorBase
 *
 * selects a subset of a track collection, copying extra information on demand
 *
 * \author Giovanni Petrucciani
 *
 * \version $Revision: 1.3 $
 *
 * $Id: TrackFullCloneSelectorBase.h,v 1.3 2010/02/11 00:10:51 wmtan Exp $
 *
 */

#include <memory>

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <algorithm>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

namespace reco {
  namespace modules {

    template <typename Selector>
    class TrackFullCloneSelectorBase : public edm::stream::EDProducer<> {
    public:
      /// constructor
      explicit TrackFullCloneSelectorBase(const edm::ParameterSet& cfg)
          : hSrcTrackToken_(consumes<reco::TrackCollection>(cfg.template getParameter<edm::InputTag>("src"))),
            hTrajToken_(mayConsume<std::vector<Trajectory> >(cfg.template getParameter<edm::InputTag>("src"))),
            hTTAssToken_(mayConsume<TrajTrackAssociationCollection>(cfg.template getParameter<edm::InputTag>("src"))),
            copyExtras_(cfg.template getUntrackedParameter<bool>("copyExtras", false)),
            copyTrajectories_(cfg.template getUntrackedParameter<bool>("copyTrajectories", false)),
            selector_(cfg, consumesCollector()) {
        std::string alias(cfg.getParameter<std::string>("@module_label"));
        produces<reco::TrackCollection>().setBranchAlias(alias + "Tracks");
        if (copyExtras_) {
          produces<reco::TrackExtraCollection>().setBranchAlias(alias + "TrackExtras");
          produces<TrackingRecHitCollection>().setBranchAlias(alias + "RecHits");
          if (copyTrajectories_) {
            produces<std::vector<Trajectory> >().setBranchAlias(alias + "Trajectories");
            produces<TrajTrackAssociationCollection>().setBranchAlias(alias + "TrajectoryTrackAssociations");
          }
        }
      }
      /// destructor
      ~TrackFullCloneSelectorBase() override {}

    private:
      /// process one event
      void produce(edm::Event& evt, const edm::EventSetup& es) override {
        edm::Handle<reco::TrackCollection> hSrcTrack;
        evt.getByToken(hSrcTrackToken_, hSrcTrack);

        selTracks_ = std::make_unique<reco::TrackCollection>();
        if (copyExtras_) {
          selTrackExtras_ = std::make_unique<reco::TrackExtraCollection>();
          selHits_ = std::make_unique<TrackingRecHitCollection>();
        }

        TrackRefProd rTracks = evt.template getRefBeforePut<TrackCollection>();

        TrackingRecHitRefProd rHits;
        TrackExtraRefProd rTrackExtras;
        if (copyExtras_) {
          rHits = evt.template getRefBeforePut<TrackingRecHitCollection>();
          rTrackExtras = evt.template getRefBeforePut<TrackExtraCollection>();
        }

        typedef reco::TrackRef::key_type TrackRefKey;
        std::map<TrackRefKey, reco::TrackRef> goodTracks;
        TrackRefKey current = 0;

        selector_.init(evt, es);
        auto tkBegin = hSrcTrack->begin();
        for (reco::TrackCollection::const_iterator it = tkBegin, ed = hSrcTrack->end(); it != ed; ++it, ++current) {
          const reco::Track& trk = *it;
          const reco::TrackRef tkref(hSrcTrack, std::distance(tkBegin, it));
          if (!selector_(tkref))
            continue;

          selTracks_->push_back(Track(trk));  // clone and store
          if (!copyExtras_)
            continue;

          // TrackExtras
          selTrackExtras_->push_back(TrackExtra(trk.outerPosition(),
                                                trk.outerMomentum(),
                                                trk.outerOk(),
                                                trk.innerPosition(),
                                                trk.innerMomentum(),
                                                trk.innerOk(),
                                                trk.outerStateCovariance(),
                                                trk.outerDetId(),
                                                trk.innerStateCovariance(),
                                                trk.innerDetId(),
                                                trk.seedDirection()));
          selTracks_->back().setExtra(TrackExtraRef(rTrackExtras, selTrackExtras_->size() - 1));
          TrackExtra& tx = selTrackExtras_->back();
          // TrackingRecHits
          auto const firstHitIndex = selHits_->size();
          for (auto const& hit : trk.recHits())
            selHits_->push_back(hit->clone());
          tx.setHits(rHits, firstHitIndex, selHits_->size() - firstHitIndex);
          tx.setTrajParams(trk.extra()->trajParams(), trk.extra()->chi2sX5());
          assert(tx.trajParams().size() == tx.recHitsSize());
          if (copyTrajectories_) {
            goodTracks[current] = reco::TrackRef(rTracks, selTracks_->size() - 1);
          }
        }
        if (copyTrajectories_) {
          edm::Handle<std::vector<Trajectory> > hTraj;
          edm::Handle<TrajTrackAssociationCollection> hTTAss;
          evt.getByToken(hTTAssToken_, hTTAss);
          evt.getByToken(hTrajToken_, hTraj);
          edm::RefProd<std::vector<Trajectory> > TrajRefProd = evt.template getRefBeforePut<std::vector<Trajectory> >();
          selTrajs_ = std::make_unique<std::vector<Trajectory> >();
          selTTAss_ = std::make_unique<TrajTrackAssociationCollection>();
          for (size_t i = 0, n = hTraj->size(); i < n; ++i) {
            edm::Ref<std::vector<Trajectory> > trajRef(hTraj, i);
            TrajTrackAssociationCollection::const_iterator match = hTTAss->find(trajRef);
            if (match != hTTAss->end()) {
              const edm::Ref<reco::TrackCollection>& trkRef = match->val;
              TrackRefKey oldKey = trkRef.key();
              std::map<TrackRefKey, reco::TrackRef>::iterator getref = goodTracks.find(oldKey);
              if (getref != goodTracks.end()) {
                // do the clone
                selTrajs_->push_back(Trajectory(*trajRef));
                selTTAss_->insert(edm::Ref<std::vector<Trajectory> >(TrajRefProd, selTrajs_->size() - 1),
                                  getref->second);
              }
            }
          }
        }

        evt.put(std::move(selTracks_));
        if (copyExtras_) {
          evt.put(std::move(selTrackExtras_));
          evt.put(std::move(selHits_));
          if (copyTrajectories_) {
            evt.put(std::move(selTrajs_));
            evt.put(std::move(selTTAss_));
          }
        }
      }
      /// source collection label
      edm::EDGetTokenT<reco::TrackCollection> hSrcTrackToken_;
      edm::EDGetTokenT<std::vector<Trajectory> > hTrajToken_;
      edm::EDGetTokenT<TrajTrackAssociationCollection> hTTAssToken_;
      /// copy only the tracks, not extras and rechits (for AOD)
      bool copyExtras_;
      /// copy also trajectories and trajectory->track associations
      bool copyTrajectories_;
      /// filter event
      Selector selector_;
      // some space
      std::unique_ptr<reco::TrackCollection> selTracks_;
      std::unique_ptr<reco::TrackExtraCollection> selTrackExtras_;
      std::unique_ptr<TrackingRecHitCollection> selHits_;
      std::unique_ptr<std::vector<Trajectory> > selTrajs_;
      std::unique_ptr<TrajTrackAssociationCollection> selTTAss_;
    };

  }  // namespace modules
}  // namespace reco
#endif
