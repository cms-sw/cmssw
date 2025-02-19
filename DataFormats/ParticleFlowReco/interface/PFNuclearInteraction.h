#ifndef _PFNuclarInteraction_H
#define _PFNuclarInteraction_H

// class which contains the secondary PFRecTracks
// this dataformat will be used to create PFBlockElementNuclTrack

// \author vincent roberfroid

#include "DataFormats/VertexReco/interface/NuclearInteraction.h"
#include "DataFormats/VertexReco/interface/NuclearInteractionFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"

namespace reco {
class PFNuclearInteraction {

  public :
   typedef NuclearInteraction::trackRef_iterator trackRef_iterator;
   typedef PFRecTrackRefVector::const_iterator   pfTrackref_iterator;

  public :
  
    PFNuclearInteraction() {}
    PFNuclearInteraction( const NuclearInteractionRef& nuclref, const PFRecTrackRefVector& pfSeconds) : nuclInterRef_(nuclref), pfSecTracks_(pfSeconds) {}

    /// \return the base reference to the primary track
    const edm::RefToBase<reco::Track>& primaryTrack() const { return nuclInterRef_->primaryTrack(); }

    /// \return first iterator over secondary tracks
    trackRef_iterator secondaryTracks_begin() const { return nuclInterRef_->secondaryTracks_begin(); }

    /// \return last iterator over secondary tracks
    trackRef_iterator secondaryTracks_end() const { return nuclInterRef_->secondaryTracks_end(); }

    /// \return first iterator over secondary PFRecTracks
    pfTrackref_iterator secPFRecTracks_begin() const { return pfSecTracks_.begin(); }

    /// \return last iterator over secondary PFRecTracks
    pfTrackref_iterator secPFRecTracks_end() const { return pfSecTracks_.end(); }
     
    /// \return the likelihood
    double likelihood() const { return nuclInterRef_->likelihood(); }

    /// \return the initial nuclear interaction
    const NuclearInteractionRef& nuclInterRef() const { return nuclInterRef_; }
    
    int secondaryTracksSize() const { return nuclInterRef_->secondaryTracksSize(); }
  private :
    // Reference to the initial NuclearInteraction
    NuclearInteractionRef nuclInterRef_;
    
    // Collection of the secondary PFRecTracks
    PFRecTrackRefVector pfSecTracks_;

 };

  /// collection of NuclearInteractions
  typedef std::vector<PFNuclearInteraction> PFNuclearInteractionCollection;
  /// persistent reference to a NuclearInteraction
  typedef edm::Ref<PFNuclearInteractionCollection> PFNuclearInteractionRef;
  /// vector of reference to Track in the same collection
  typedef edm::RefVector<PFNuclearInteractionCollection> PFNuclearInteractionRefVector;
}
#endif
