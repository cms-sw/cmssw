#ifndef ConvBremSeed_h
#define ConvBremSeed_h 1

/** \class reco::ConvBremSeed
 *
 * ConvBremSeed is a seed object constructed from a supercluster and 2 PixelRecHits
 *
 * \author M.Pioppi CERN
 *
 * \version   1st Version Oct 6, 2008  

 *
 ************************************************************/

#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrackFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/Common/interface/Ref.h"

#include <vector>

namespace reco {


class ConvBremSeed: public TrajectorySeed
 {
  public :
        
    typedef edm::OwnVector<TrackingRecHit> recHitContainer;


    ConvBremSeed(){} 
    ~ConvBremSeed() override {}
   

    /// Constructor from TrajectorySeed
    ConvBremSeed( const TrajectorySeed & seed,edm::Ref<GsfPFRecTrackCollection> & pfgsf):
      TrajectorySeed(seed), pfGsf_ (pfgsf){}

    /// reference to the GSDPFRecTrack
 
    GsfPFRecTrackRef GsfPFTrack() const {return pfGsf_;}
    


    ConvBremSeed * clone() const override {return new ConvBremSeed( * this); }

 private:
    
    //! Pointer to the electromagnetic super cluster.
    GsfPFRecTrackRef  pfGsf_;

  } ;


// Class ConvBremSeed

}// namespace reco

#endif
