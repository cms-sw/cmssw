#ifndef RecoAlgos_TrackFullCloneSelectorBase_h
#define RecoAlgos_TrackFullCloneSelectorBase_h
/** \class TrackFullCloneSelectorBase
 *
 * selects a subset of a track collection, copying extra information on demand
 * 
 * \author Giovanni Petrucciani 
 *
 * \version $Revision: 1.4 $
 *
 * $Id: TrackFullCloneSelectorBase.h,v 1.4 2013/02/28 00:12:51 wmtan Exp $
 *
 */

#include <utility>
#include <vector>
#include <memory>
#include <algorithm>
#include <map>
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"


namespace reco { namespace modules {

template<typename Selector>
class TrackFullCloneSelectorBase : public edm::EDProducer {
public:
  /// constructor 
  explicit TrackFullCloneSelectorBase( const edm::ParameterSet & cfg ) :
    src_( cfg.template getParameter<edm::InputTag>( "src" ) ),
    copyExtras_(cfg.template getUntrackedParameter<bool>("copyExtras", false)),
    copyTrajectories_(cfg.template getUntrackedParameter<bool>("copyTrajectories", false)),
    selector_( cfg ) {
      std::string alias( cfg.getParameter<std::string>( "@module_label" ) );
      produces<reco::TrackCollection>().setBranchAlias( alias + "Tracks" );
      if (copyExtras_) {
          produces<reco::TrackExtraCollection>().setBranchAlias( alias + "TrackExtras" );
          produces<TrackingRecHitCollection>().setBranchAlias( alias + "RecHits" );
          if (copyTrajectories_) {
              produces< std::vector<Trajectory> >().setBranchAlias( alias + "Trajectories" );
              produces< TrajTrackAssociationCollection >().setBranchAlias( alias + "TrajectoryTrackAssociations" );
          }
      }
   }
  /// destructor
  virtual ~TrackFullCloneSelectorBase() { }
  
private:
  /// process one event
  void produce( edm::Event& evt, const edm::EventSetup& es) override {
      edm::Handle<reco::TrackCollection> hSrcTrack;
      edm::Handle< std::vector<Trajectory> > hTraj;
      edm::Handle< TrajTrackAssociationCollection > hTTAss;
      evt.getByLabel( src_, hSrcTrack );

      selTracks_ = std::auto_ptr<reco::TrackCollection>(new reco::TrackCollection());
      if (copyExtras_) {
          selTrackExtras_ = std::auto_ptr<reco::TrackExtraCollection>(new reco::TrackExtraCollection());
          selHits_ = std::auto_ptr<TrackingRecHitCollection>(new TrackingRecHitCollection());
      }

      TrackRefProd rTracks = evt.template getRefBeforePut<TrackCollection>();      

      TrackingRecHitRefProd rHits;
      TrackExtraRefProd rTrackExtras;
      if (copyExtras_) {
          rHits = evt.template getRefBeforePut<TrackingRecHitCollection>();
          rTrackExtras = evt.template getRefBeforePut<TrackExtraCollection>();
      }

      typedef reco::TrackRef::key_type TrackRefKey;
      std::map<TrackRefKey, reco::TrackRef  > goodTracks;
      TrackRefKey current = 0;

      for (reco::TrackCollection::const_iterator it = hSrcTrack->begin(), ed = hSrcTrack->end(); it != ed; ++it, ++current) {
          const reco::Track & trk = * it;
          if (!selector_(trk, evt)) continue;

          selTracks_->push_back( Track( trk ) ); // clone and store
          if (!copyExtras_) continue;

          // TrackExtras
          selTrackExtras_->push_back( TrackExtra( trk.outerPosition(), trk.outerMomentum(), trk.outerOk(),
                      trk.innerPosition(), trk.innerMomentum(), trk.innerOk(),
                      trk.outerStateCovariance(), trk.outerDetId(),
                      trk.innerStateCovariance(), trk.innerDetId(),
                      trk.seedDirection() ) );
          selTracks_->back().setExtra( TrackExtraRef( rTrackExtras, selTrackExtras_->size() - 1) );
          TrackExtra & tx = selTrackExtras_->back();
          // TrackingRecHits
          for( trackingRecHit_iterator hit = trk.recHitsBegin(); hit != trk.recHitsEnd(); ++ hit ) {
              selHits_->push_back( (*hit)->clone() );
              tx.add( TrackingRecHitRef( rHits, selHits_->size() - 1) );
          }
          if (copyTrajectories_) {
              goodTracks[current] = reco::TrackRef(rTracks, selTracks_->size() - 1);
          }
      }
      if ( copyTrajectories_ ) {
          edm::Handle< std::vector<Trajectory> > hTraj;
          edm::Handle< TrajTrackAssociationCollection > hTTAss;
          evt.getByLabel(src_, hTTAss);
          evt.getByLabel(src_, hTraj);
          edm::RefProd< std::vector<Trajectory> > TrajRefProd = evt.template getRefBeforePut< std::vector<Trajectory> >();
          selTrajs_ = std::auto_ptr< std::vector<Trajectory> >(new std::vector<Trajectory>()); 
          selTTAss_ = std::auto_ptr< TrajTrackAssociationCollection >(new TrajTrackAssociationCollection());
          for (size_t i = 0, n = hTraj->size(); i < n; ++i) {
              edm::Ref< std::vector<Trajectory> > trajRef(hTraj, i);
              TrajTrackAssociationCollection::const_iterator match = hTTAss->find(trajRef);
              if (match != hTTAss->end()) {
                  const edm::Ref<reco::TrackCollection> &trkRef = match->val; 
                  TrackRefKey oldKey = trkRef.key();
                  std::map<TrackRefKey, reco::TrackRef>::iterator getref = goodTracks.find(oldKey);        
                  if (getref != goodTracks.end()) {
                      // do the clone
                      selTrajs_->push_back( Trajectory(*trajRef) );
                      selTTAss_->insert ( edm::Ref< std::vector<Trajectory> >(TrajRefProd, selTrajs_->size() - 1),
                                          getref->second );
                  }
              }
          }
      }
      
      evt.put(selTracks_);
      if (copyExtras_) {
            evt.put(selTrackExtras_); 
            evt.put(selHits_);
            if ( copyTrajectories_ ) {
                evt.put(selTrajs_);
                evt.put(selTTAss_);
            }
      }
  }
  /// source collection label
  edm::InputTag src_;
  /// copy only the tracks, not extras and rechits (for AOD)
  bool copyExtras_;
  /// copy also trajectories and trajectory->track associations
  bool copyTrajectories_;
  /// filter event
  Selector selector_;
  // some space
  std::auto_ptr<reco::TrackCollection> selTracks_;
  std::auto_ptr<reco::TrackExtraCollection> selTrackExtras_;
  std::auto_ptr<TrackingRecHitCollection> selHits_;
  std::auto_ptr< std::vector<Trajectory> > selTrajs_;
  std::auto_ptr< TrajTrackAssociationCollection > selTTAss_;
};

} }
#endif
