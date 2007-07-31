#ifndef DataFormats_EgammaReco_PreshowerCluster_h
#define DataFormats_EgammaReco_PreshowerCluster_h
/*
 * Preshower cluster class
 *
 * \authors Dmirty Bandurin (KSU), Ted Kolberg (ND)
 */
// $Id: PreshowerCluster.h,v 1.15 2007/02/14 15:45:10 futyand Exp $
//
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h" 

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
                     const int plane);

    /// Constructor from cluster
    PreshowerCluster(const PreshowerCluster&);

    /// Number of RecHits the cluster
    int nhits() const {return usedHits_.size();}

    /// Preshower plane
    int plane() const { return plane_; }

    double et() const { return energy()/cosh(eta()); }

    /// Comparisons
    bool operator==(const PreshowerCluster&) const;
    bool operator<(const PreshowerCluster&) const;

    /// Associated basic cluster;
    BasicClusterRef basicCluster() const {return bc_ref_;}

    /// DetIds of component RecHits
    virtual std::vector<DetId> getHitsByDetId() const { return usedHits_; }

    void setBCRef( const BasicClusterRef & r ) { bc_ref_ = r; }

  private:

    int plane_;

    /// Associated basic cluster;
    BasicClusterRef bc_ref_;

    /// used hits by detId
    std::vector<DetId> usedHits_;
  };
}
#endif
