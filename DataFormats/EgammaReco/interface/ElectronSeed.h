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

#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
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

    static std::string const & name()
     {
      static std::string const name_("ElectronSeed") ;
      return name_;
     }

    //! Construction of base attributes
    ElectronSeed() ;
    ElectronSeed( const TrajectorySeed & ) ;
    ElectronSeed( PTrajectoryStateOnDet & pts, RecHitContainer & rh,  PropagationDirection & dir ) ;
    ElectronSeed * clone() const { return new ElectronSeed(*this) ; }
    virtual ~ElectronSeed() ;

    //! Set additional info
    void setCtfTrack( const CtfTrackRef & ) ;
    void setCaloCluster
     ( const CaloClusterRef &,
       unsigned char hitsMask =0,
       int subDet2 =0, int subDet1 =0,
       float hoe1 =std::numeric_limits<float>::infinity(),
       float hoe2 =std::numeric_limits<float>::infinity() ) ;
    void setNegAttributes
     ( float dRz2 =std::numeric_limits<float>::infinity(),
       float dPhi2 =std::numeric_limits<float>::infinity(),
       float dRz1 =std::numeric_limits<float>::infinity(),
       float dPhi1 =std::numeric_limits<float>::infinity() ) ;
    void setPosAttributes
     ( float dRz2 =std::numeric_limits<float>::infinity(),
       float dPhi2 =std::numeric_limits<float>::infinity(),
       float dRz1 =std::numeric_limits<float>::infinity(),
       float dPhi1 =std::numeric_limits<float>::infinity() ) ;

    //! Accessors
    const CtfTrackRef& ctfTrack() const { return ctfTrack_ ; }
    const CaloClusterRef& caloCluster() const { return caloCluster_ ; }
    unsigned char hitsMask() const { return hitsMask_ ; }
    int subDet2() const { return subDet2_ ; }
    float dRz2() const { return dRz2_ ; }
    float dPhi2() const { return dPhi2_ ; }
    float dRz2Pos() const { return dRz2Pos_ ; }
    float dPhi2Pos() const { return dPhi2Pos_ ; }
    int subDet1() const { return subDet1_ ; }
    float dRz1() const { return dRz1_ ; }
    float dPhi1() const { return dPhi1_ ; }
    float dRz1Pos() const { return dRz1Pos_ ; }
    float dPhi1Pos() const { return dPhi1Pos_ ; }
    float hoe1() const { return hcalDepth1OverEcal_ ; }
    float hoe2() const { return hcalDepth2OverEcal_ ; }

    //! Utility
    TrackCharge getCharge() const { return startingState().parameters().charge() ; }

    bool isEcalDriven() const { return isEcalDriven_ ; }
    bool isTrackerDriven() const { return isTrackerDriven_ ; }

  private:

    CtfTrackRef ctfTrack_ ;
    CaloClusterRef caloCluster_ ;
    unsigned char hitsMask_ ;
    int subDet2_ ;
    float dRz2_ ;
    float dPhi2_ ;
    float dRz2Pos_ ;
    float dPhi2Pos_ ;
    int subDet1_ ;
    float dRz1_ ;
    float dPhi1_ ;
    float dRz1Pos_ ;
    float dPhi1Pos_ ;
    float hcalDepth1OverEcal_ ; // hcal over ecal seed cluster energy using first hcal depth
    float hcalDepth2OverEcal_ ; // hcal over ecal seed cluster energy using 2nd hcal depth
    bool isEcalDriven_ ;
    bool isTrackerDriven_ ;

 } ;

 }

#endif
