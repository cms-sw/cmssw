#ifndef DataFormats_EgammaReco_PreshowerCluster_h
#define DataFormats_EgammaReco_PreshowerCluster_h
/*
 * Preshower cluster class
 *
 * \authors Dmirty Bandurin (KSU), Ted Kolberg (ND)
 */
// $Id: PreshowerCluster.h,v 1.11 2006/07/21 14:02:04 rahatlou Exp $
//
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EgammaReco/interface/PreshowerClusterFwd.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"

#include <cmath>

namespace reco {

  class PreshowerCluster : public EcalCluster {
  public:

    typedef math::XYZPoint Point;

    /// default constructor
    PreshowerCluster() : EcalCluster(0., Point(0.,0.,0.)) { };

    virtual ~PreshowerCluster();

    /// Constructor from EcalRecHits
    PreshowerCluster(const double E, const Point& pos, 
                     const std::vector<DetId> usedHits, 
                     reco::BasicClusterRefVector::iterator BC_ref, 
                     const int plane);

    /// Constructor from cluster
    PreshowerCluster(const PreshowerCluster&);

    /// Number of RecHits the cluster
    int nhits() const {return usedHits_.size();}

    /// Preshower plane
    int plane() {
      return plane_;
    }

    double et() const {
      return energy()/cosh(eta());
    }

    /// Comparisons
    bool operator==(const PreshowerCluster&) const;
    bool operator<(const PreshowerCluster&) const;

    /// Associated basic cluster;
    BasicClusterRef basicCluster() const {return bc_ref_;}

    /// DetIds of component RecHits
    virtual std::vector<DetId> getHitsByDetId() const { return usedHits_; }

  private:

    int plane_;

    /// Associated basic cluster;
    BasicClusterRef bc_ref_;

    /// used hits by detId
    std::vector<DetId> usedHits_;
  };
}
#endif
