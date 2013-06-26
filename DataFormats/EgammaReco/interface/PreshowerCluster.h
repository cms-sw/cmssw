#ifndef DataFormats_EgammaReco_PreshowerCluster_h
#define DataFormats_EgammaReco_PreshowerCluster_h
/*
 * Preshower cluster class
 *
 * \authors Dmirty Bandurin (KSU), Ted Kolberg (ND)
 */
// $Id: PreshowerCluster.h,v 1.20 2013/04/22 22:53:02 wmtan Exp $
//
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"

#include <cmath>

namespace reco {

  class PreshowerCluster : public CaloCluster {
  public:

    typedef math::XYZPoint Point;

    /// default constructor
    PreshowerCluster() : CaloCluster(0., Point(0.,0.,0.)) { };

    virtual ~PreshowerCluster();

    /// Constructor from EcalRecHits
    PreshowerCluster(const double E, const Point& pos, 
                     const std::vector< std::pair<DetId, float> >& usedHits, 
                     const int plane);

    /// Constructor from cluster
    PreshowerCluster(const PreshowerCluster&);

    /// Number of RecHits the cluster
    int nhits() const {return hitsAndFractions_.size();}

    /// Preshower plane
    int plane() const { return plane_; }

    double et() const { return energy()/cosh(eta()); }

    /// Comparisons
    bool operator==(const PreshowerCluster&) const;
    bool operator<(const PreshowerCluster&) const;

    /// Associated basic cluster;
    CaloClusterPtr basicCluster() const {return bc_ref_;}

    /// DetIds of component RecHits -- now inherited from CaloCluster
    //std::vector<DetId> getHitsByDetId() const { return usedHits_; }

    void setBCRef( const CaloClusterPtr & r ) { bc_ref_ = r; }

  private:

    int plane_;

    /// Associated basic cluster;
    CaloClusterPtr bc_ref_;

    /// used hits by detId -- now inherited from CaloCluster
    //std::vector<DetId> usedHits_;
  };
}
#endif
