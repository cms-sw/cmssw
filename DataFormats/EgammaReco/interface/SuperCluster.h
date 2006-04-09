#ifndef EgammaReco_SuperCluster_h
#define EgammaReco_SuperCluster_h
/** \class reco::SuperCluster
 *  
 * A SuperCluster reconstructed in the Electromagnetic Calorimeter
 * contains references to seed and constituent BasicClusters
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: SuperCluster.h,v 1.10 2006/03/20 14:06:37 llista Exp $
 *
 */
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"
#include "DataFormats/EgammaReco/interface/ClusterPi0DiscriminatorFwd.h"

namespace reco {
  class SuperCluster {
  public:
    /// a spatial vector
    typedef math::RhoEtaPhiVector Vector;
    /// a point in the space
    typedef math::XYZPoint Point;
    /// default constructor
    SuperCluster() { }
    /// constructor from values
    SuperCluster( const Vector &, const Point &, double uE );
    /// momentum vector
    const Vector & momentum() const { return momentum_; }
    /// cluster energy
    double energy() const { return momentum_.R(); }
    /// cluster uncorrected energy
    double uncorrectedEnergy() const { return uncorrectedEnergy_; }
    /// cluster centroid pseudorapidity
    double eta() const { return momentum_.Eta(); }
    /// cluster centroid azimuthal angle
    double phi() const { return momentum_.Phi(); }
    /// cluster centroid polar angle
    double theta() const { return momentum_.Theta(); }
    /// cluster centroid position
    const Point & position() const { return position_; }
    /// x coordinate of cluster centroid
    double x() const { return position_.X(); }
    /// y coordinate of cluster centroid
    double y() const { return position_.Y(); }
    /// z coordinate of cluster centroid
    double z() const { return position_.Z(); }
    /// polar radius of cluster centroid
    double rho() const { return position_.Rho(); }
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
    /// ratio of energy deposits in Hcal over Ecal
    double hadOverEcal() const;
    /// reference to pi0 discriminator information
    const ClusterPi0DiscriminatorRef & pi0Discriminator() const { return pi0Discriminator_; }
    /// set reference to pi0 discriminator information
    void setPi0Discriminator( const ClusterPi0DiscriminatorRef & ref ) {
      pi0Discriminator_ = ref;
    }
    /// pi0 discriminator variable #1 (should be better documented!)
    double disc1() const;
    /// pi0 discriminator variable #2 (should be better documented!)
    double disc2() const;
    /// pi0 discriminator variable #3 (should be better documented!)
    double disc3() const;

  private:
    /// momentum vector
    Vector momentum_;
    /// position
    Point position_;
    /// uncorrected energy
    Double32_t uncorrectedEnergy_;
    /// reference to BasicCluster seed
    BasicClusterRef seed_;
    /// references to BasicCluster constitunets
    BasicClusterRefVector clusters_;
    /// refrence to shape variable information
    ClusterShapeRef shape_;
    /// reference to pi0 discriminator information
    ClusterPi0DiscriminatorRef pi0Discriminator_;
  };

}

#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#endif
