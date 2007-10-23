#ifndef ElectronPixelSeed_h
#define ElectronPixelSeed_h 1

/** \class reco::ElectronPixelSeed
 *
 * ElectronPixelSeed is a seed object constructed from a supercluster and 2 PixelRecHits
 *
 * \author U.Berthon, C.Charlot, LLR Palaiseau
 *
 * \version   1st Version May 30, 2006  

 *
 ************************************************************/

#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectoryState/interface/TrackCharge.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/Common/interface/Ref.h"

#include <vector>

namespace reco {


class ElectronPixelSeed: public TrajectorySeed
 {
  public :
        
    typedef edm::OwnVector<TrackingRecHit> recHitContainer;
    static std::string const &name() 
    { 
      static std::string const name_("ElectronPixelSeed");
      return name_;
    }

    ElectronPixelSeed() ;
    ElectronPixelSeed( const ElectronPixelSeed & ) ;
    ElectronPixelSeed & operator=( const ElectronPixelSeed & ) ;
    virtual ~ElectronPixelSeed() ;
   
    //! Constructor from two hits
    ElectronPixelSeed(edm::Ref<SuperClusterCollection> & seed, PTrajectoryStateOnDet & pts, recHitContainer & rh,  PropagationDirection & dir);

    // Constructor from TrajectorySeed
    ElectronPixelSeed(edm::Ref<SuperClusterCollection> & scl, const TrajectorySeed & seed) ;

    // 
    SuperClusterRef superCluster() const {return theSuperCluster; }
    
    // interfaces

    TrackCharge getCharge() const {return startingState().parameters().charge();}

    ElectronPixelSeed * clone() const {return new ElectronPixelSeed( * this); }

 private:
    
    //! Pointer to the electromagnetic super cluster.
    SuperClusterRef theSuperCluster;

  } ;


// Class ElectronPixelSeed

}// namespace reco

#endif
