#ifndef EgammaReco_SuperCluster_h
#define EgammaReco_SuperCluster_h
/** \class reco::SuperCluster SuperCluster.h DataFormats/EgammaReco/interface/SuperCluster.h
 *  
 * A SuperCluster reconstructed in the Electromagnetic Calorimeter
 * contains references to seed and constituent BasicClusters
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: SuperCluster.h,v 1.6 2006/05/23 16:26:23 askew Exp $
 *
 */
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"
#include "DataFormats/EgammaReco/interface/EcalCluster.h"
#include "DataFormats/DetId/interface/DetId.h"

namespace reco {
  class SuperCluster : public EcalCluster {
  public:
    /// a spatial vector
    typedef math::RhoEtaPhiVector Vector;
    /// a point in the space
    typedef math::XYZPoint Point;

    /// default constructor
    SuperCluster() : EcalCluster(0., Point(0.,0.,0.)) { }

    /// constructor from values
    //SuperCluster( const Vector &, const Point &, double uE );

    /// constructor defined by EcalCluster - will have to use setSedd and add() separately
    SuperCluster( double energy, const math::XYZPoint& position );

    SuperCluster( double energy, const math::XYZPoint& position,
                  const BasicClusterRef & seed,
                  const BasicClusterRefVector& clusters);



    /// x coordinate of cluster centroid
    double x() const { return position().X(); }

    /// y coordinate of cluster centroid
    double y() const { return position().Y(); }

    /// z coordinate of cluster centroid
    double z() const { return position().Z(); }

    /// polar radius of cluster centroid
    double rho() const { return position().Rho(); }

    /// seed BasicCluster
    const BasicClusterRef & seed() const { return seed_; }

    /// set reference to seed BasicCluster
    void setSeed( const BasicClusterRef & r ) { seed_ = r; }

    /// add reference to constituent BasicCluster
    void add( const BasicClusterRef & r ) { clusters_.push_back( r ); }

    /// fist iterator over BasicCluster constituents
    basicCluster_iterator clustersBegin() const { return clusters_.begin(); }

    /// last iterator over BasicCluster constituents
    basicCluster_iterator clustersEnd() const { return clusters_.end(); }

    /// number of BasicCluster constituents
    size_t clustersSize() const { return clusters_.size(); }

    /// list of used xtals by DetId
    virtual std::vector<DetId> getHitsByDetId() const;

    /// reference 
    const ClusterShapeRef & shape() const { return shape_; }
    void setShape( const ClusterShapeRef & ref ) { 
      shape_ = ref; 
    }

    /// maximum energy in a single crystal
    double eMax() const;
    /// energy in the most energetic 2x2 block
    double e2x2() const;
    /// energy in the most energetic 3x3 block
    double e3x3() const;
    /// energy in the most energetic 5x5 block
    double e5x5() const;
    /// covariance element in pseudorapidity
    double covEtaEta() const;
    /// covariance element in pseudorapidity - phi
    double covEtaPhi() const;
    /// covariance element in pseudorapidity
    double covPhiPhi() const;

  private:

    /// reference to BasicCluster seed
    BasicClusterRef seed_;

    /// references to BasicCluster constitunets
    BasicClusterRefVector clusters_;

    /// refrence to shape variable information
    ClusterShapeRef shape_;

   /// used hits by detId - retrieved from BC constituents
   std::vector<DetId> usedHits_;

  };

}
#endif
