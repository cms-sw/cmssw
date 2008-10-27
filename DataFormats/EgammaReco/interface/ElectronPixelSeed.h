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
#include <limits>

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
    //DC: DEFAULT ONES ARE GOOD ENOUGH
    //ElectronPixelSeed( const ElectronPixelSeed & ) ;
    //ElectronPixelSeed & operator=( const ElectronPixelSeed & ) ;
    virtual ~ElectronPixelSeed() ;
   
    //! Constructor from two hits
    ElectronPixelSeed( edm::Ref<SuperClusterCollection> & seed, PTrajectoryStateOnDet & pts, recHitContainer & rh,  PropagationDirection & dir,
      int subDet2 =0, float dRz2 =std::numeric_limits<float>::infinity(), float dPhi2 =std::numeric_limits<float>::infinity() ) ;

    // Constructor from TrajectorySeed
    ElectronPixelSeed( edm::Ref<SuperClusterCollection> & scl, const TrajectorySeed & seed,
      int subDet2 =0, float dRz2 =std::numeric_limits<float>::infinity(), float dPhi2 =std::numeric_limits<float>::infinity() ) ;

    // additionnal info
    SuperClusterRef superCluster() const { return theSuperCluster ; }
    int subDet2() const { return subDet2_ ; }
    float dRz2() const { return dRz2_ ; }
    float dPhi2() const { return dPhi2_ ; }
   
    // interfaces

    TrackCharge getCharge() const {return startingState().parameters().charge();}

    ElectronPixelSeed * clone() const {return new ElectronPixelSeed( * this); }

 private:
    
    //! Pointer to the electromagnetic super cluster.
    SuperClusterRef theSuperCluster;
    int subDet2_ ;
    float dRz2_ ;
    float dPhi2_ ;

  } ;


// Class ElectronPixelSeed

}// namespace reco

#endif
