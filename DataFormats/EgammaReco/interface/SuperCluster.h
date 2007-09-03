#ifndef EgammaReco_SuperCluster_h
#define EgammaReco_SuperCluster_h
/** \class reco::SuperCluster SuperCluster.h DataFormats/EgammaReco/interface/SuperCluster.h
 *  
 * A SuperCluster reconstructed in the Electromagnetic Calorimeter
 * contains references to seed and constituent BasicClusters
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: SuperCluster.h,v 1.10 2007/07/31 15:20:04 ratnik Exp $
 *
 */
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/EcalCluster.h"
#include "DataFormats/DetId/interface/DetId.h"

namespace reco {
  class SuperCluster : public EcalCluster {
  public:

    typedef math::XYZPoint Point;

    /// default constructor
    SuperCluster() : EcalCluster(0., Point(0.,0.,0.)), rawEnergy_(-1.) {}

    /// constructor defined by EcalCluster - will have to use setSeed and add() separately
    SuperCluster( double energy, const Point& position );

    SuperCluster( double energy, const Point& position,
                  const BasicClusterRef & seed,
                  const BasicClusterRefVector& clusters,
		  double Epreshower = 0.);

    /// raw uncorrected energy (sum of energies of component BasicClusters)
    double rawEnergy() const;

    /// energy deposited in preshower 
    double preshowerEnergy() const { return preshowerEnergy_; }

    /// seed BasicCluster
    const BasicClusterRef & seed() const { return seed_; }

    /// fist iterator over BasicCluster constituents
    basicCluster_iterator clustersBegin() const { return clusters_.begin(); }

    /// last iterator over BasicCluster constituents
    basicCluster_iterator clustersEnd() const { return clusters_.end(); }

    /// number of BasicCluster constituents
    size_t clustersSize() const { return clusters_.size(); }

    /// list of used xtals by DetId
    virtual std::vector<DetId> getHitsByDetId() const { return usedHits_; }

    /// set reference to seed BasicCluster
    void setSeed( const BasicClusterRef & r ) { seed_ = r; }

    /// add reference to constituent BasicCluster
    void add( const BasicClusterRef & r ) { clusters_.push_back( r ); }

  private:

    /// reference to BasicCluster seed
    BasicClusterRef seed_;

    /// references to BasicCluster constitunets
    BasicClusterRefVector clusters_;

    /// used hits by detId - retrieved from BC constituents
    std::vector<DetId> usedHits_;

    double preshowerEnergy_;

    mutable double rawEnergy_;

  };

}
#endif
