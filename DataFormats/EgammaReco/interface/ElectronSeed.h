#ifndef ElectronSeed_h
#define ElectronSeed_h 1

/** \class reco::ElectronSeed
 *
 * ElectronSeed is a seed for gsf tracking,  constructed from
 * either a supercluster or a ctf track.
 *
 * \author D.Chamont, U.Berthon, C.Charlot, LLR Palaiseau
 *
 ************************************************************/

#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectoryState/interface/TrackCharge.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/Ref.h"

#include <vector>
#include <limits>

namespace reco
 {

class ElectronSeed : public TrajectorySeed
 {
  public :

    typedef edm::OwnVector<TrackingRecHit> RecHitContainer ;
    typedef edm::RefToBase<CaloCluster> CaloClusterRef ;
    typedef edm::Ref<TrackCollection> CtfTrackRef ;

    static std::string const &name()
    {
      static std::string const name_("ElectronSeed");
      return name_;
    }

    ElectronSeed() ;
    virtual ~ElectronSeed() ;

    //! Tracker driven
    ElectronSeed( const TrajectorySeed &, const CtfTrackRef & ) ;

    //! Ecal driven
    ElectronSeed( const TrajectorySeed &, const CaloClusterRef &,
      int subDet2 =0, float dRz2 =std::numeric_limits<float>::infinity(),
      float dPhi2 =std::numeric_limits<float>::infinity() ) ;

    //! Ecal driven, old seeding scheme
    ElectronSeed( PTrajectoryStateOnDet & pts, RecHitContainer & rh,  PropagationDirection & dir,
      const CaloClusterRef &,
      int subDet2 =0, float dRz2 =std::numeric_limits<float>::infinity(),
      float dPhi2 =std::numeric_limits<float>::infinity() ) ;

    //! Accessors
    CaloClusterRef caloCluster() const { return caloCluster_ ; }
    CtfTrackRef ctfTrack() const { return ctfTrack_ ; }
    int subDet2() const { return subDet2_ ; }
    float dRz2() const { return dRz2_ ; }
    float dPhi2() const { return dPhi2_ ; }
    TrackCharge getCharge() const { return startingState().parameters().charge() ; }
    ElectronSeed * clone() const { return new ElectronSeed(*this) ; }

  private:

    CtfTrackRef ctfTrack_ ;
    CaloClusterRef caloCluster_ ;
    int subDet2_ ;
    float dRz2_ ;
    float dPhi2_ ;

 } ;

 }

#endif
