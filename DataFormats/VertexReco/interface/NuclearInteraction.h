#ifndef DATAFORMAT_NUCLEARINTERACTION_
#define DATAFORMAT_NUCLEARINTERACTION_

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

namespace reco {
  
   class NuclearInteraction {
     
     public :

       typedef edm::RefVector<TrajectorySeedCollection>  TrajectorySeedRefVector;
       typedef edm::Ref<TrajectorySeedCollection>        TrajectorySeedRef;
       typedef reco::Vertex::trackRef_iterator           trackRef_iterator;
       typedef TrajectorySeedRefVector::iterator         seedRef_iterator;

       NuclearInteraction() {}

       NuclearInteraction( const TrajectorySeedRefVector& tseeds, const reco::Vertex& vtx, double lkh) { 
                         seeds_ = tseeds;
                         vertex_ = vtx;
                         likelihood_ = lkh;
       }
 
       /// return the base reference to the primary track
       const edm::RefToBase<reco::Track>& primaryTrack() const { return *(vertex_.tracks_begin()); }

       /// return the number of secondary tracks
       int secondaryTracksSize() const { return vertex_.tracksSize()-1; }

       /// first iterator over secondary tracks
       trackRef_iterator secondaryTracks_begin() const { return vertex_.tracks_begin()+1; }

       /// last iterator over secondary tracks
       trackRef_iterator secondaryTracks_end() const { return vertex_.tracks_end(); }

       /// return the number of seeds
       int seedsSize() const { return seeds_.size(); }

       /// return the seeds
       const TrajectorySeedRefVector& seeds() { return seeds_; }

       /// first iterator over seeds
       seedRef_iterator seeds_begin() const { return seeds_.begin(); }

       /// last iterator over seeds
       seedRef_iterator seeds_end() const { return seeds_.end(); }

       /// return the vertex
       const reco::Vertex& vertex() const { return vertex_; }

       /// return the likelihood ~ probability that the vertex is a real nuclear interaction
       double likelihood() const { return likelihood_; }

     private :

        /// The refitted primary track after removing eventually some outer rechits
        //reco::Track                 refittedPrimaryTrack_; // to be included in a futur version

        /// Reference to the TrajectorySeeds produced by NuclearSeedGenerator
        TrajectorySeedRefVector     seeds_;

        /// The calculated vertex position
        reco::Vertex                vertex_;
   
        /// Varaible used to measure the quality of the reconstructed nuclear interaction
        double                      likelihood_;
     };
}
#endif
