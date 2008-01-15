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
  
    PFNuclearInteraction() {}
    PFNuclearInteraction( const NuclearInteractionRef& nuclref, const PFRecTrackCollection& pfSeconds) : nuclInterRef_(nuclref), pfSecTracks_(pfSeconds) {}
    
  private :
    // Reference to the initial NuclearInteraction
    NuclearInteractionRef nuclInterRef_;
    
    // Collection of the secondary PFRecTracks
    PFRecTrackCollection pfSecTracks_;

 };

  /// collection of NuclearInteractions
  typedef std::vector<PFNuclearInteraction> PFNuclearInteractionCollection;
  /// persistent reference to a NuclearInteraction
  typedef edm::Ref<PFNuclearInteractionCollection> PFNuclearInteractionRef;
  /// vector of reference to Track in the same collection
  typedef edm::RefVector<PFNuclearInteractionCollection> PFNuclearInteractionRefVector;
}
#endif
