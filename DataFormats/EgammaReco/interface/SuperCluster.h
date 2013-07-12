#ifndef EgammaReco_SuperCluster_h
#define EgammaReco_SuperCluster_h
/** \class reco::SuperCluster SuperCluster.h DataFormats/EgammaReco/interface/SuperCluster.h
 *  
 * A SuperCluster reconstructed in the Electromagnetic Calorimeter
 * contains references to seed and constituent BasicClusters
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: SuperCluster.h,v 1.25 2012/10/22 20:23:27 argiro Exp $
 *
 */
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/DetId/interface/DetId.h"


namespace reco {
  class SuperCluster : public CaloCluster {
  public:

    typedef math::XYZPoint Point;

    /// default constructor
    SuperCluster() : CaloCluster(0., Point(0.,0.,0.)), preshowerEnergy_(0), rawEnergy_(-1.), phiWidth_(0), etaWidth_(0), preshowerEnergy1_(0), preshowerEnergy2_(0) {}

    /// constructor defined by CaloCluster - will have to use setSeed and add() separately
    SuperCluster( double energy, const Point& position );

    SuperCluster( double energy, const Point& position,
                  const CaloClusterPtr & seed,
                  const CaloClusterPtrVector& clusters,
		  double Epreshower=0.,
		  double phiWidth=0., double etaWidth=0.,
		  double Epreshower1=0., double Epreshower2=0.);

    // to be merged in the previous one? -- FIXME
    SuperCluster( double energy, const Point& position,
                  const CaloClusterPtr & seed,
                  const CaloClusterPtrVector& clusters,
                  const CaloClusterPtrVector& preshowerClusters,
		  double Epreshower=0.,
		  double phiWidth=0., double etaWidth=0.,
		  double Epreshower1=0., double Epreshower2=0.);


    /// raw uncorrected energy (sum of energies of component BasicClusters)
    double rawEnergy() const { return rawEnergy_; }

    /// energy deposited in preshower 
    double preshowerEnergy() const { return preshowerEnergy_; }
    double preshowerEnergyPlane1() const {return preshowerEnergy1_; }
    double preshowerEnergyPlane2() const {return preshowerEnergy2_; }

    /// obtain phi and eta width of the Super Cluster
    double phiWidth() const { return phiWidth_; }
    double etaWidth() const { return etaWidth_; }

    //Assign new variables to supercluster
    void setPreshowerEnergy( double preshowerEnergy ) { preshowerEnergy_ = preshowerEnergy; };
    void setPreshowerEnergyPlane1( double preshowerEnergy1 ) { preshowerEnergy1_ = preshowerEnergy1; }; 
    void setPreshowerEnergyPlane2( double preshowerEnergy2 ) { preshowerEnergy2_ = preshowerEnergy2; }; 
    void setPhiWidth( double pw ) { phiWidth_ = pw; }
    void setEtaWidth( double ew ) { etaWidth_ = ew; }

    /// seed BasicCluster
    const CaloClusterPtr & seed() const { return seed_; }

    /// fist iterator over BasicCluster constituents
    CaloCluster_iterator clustersBegin() const { return clusters_.begin(); }

    /// last iterator over BasicCluster constituents
    CaloCluster_iterator clustersEnd() const { return clusters_.end(); }

    /// fist iterator over PreshowerCluster constituents
    CaloCluster_iterator preshowerClustersBegin() const { return preshowerClusters_.begin(); }

    /// last iterator over PreshowerCluster constituents
    CaloCluster_iterator preshowerClustersEnd() const { return preshowerClusters_.end(); }

    /// number of BasicCluster constituents
    size_t clustersSize() const { return clusters_.size(); }

    /// list of used xtals by DetId // now inherited by CaloCluster
    //std::vector<DetId> getHitsByDetId() const { return usedHits_; }

    /// set reference to seed BasicCluster
    void setSeed( const CaloClusterPtr & r ) { seed_ = r; }

    /// add reference to constituent BasicCluster
    void addCluster( const CaloClusterPtr & r ) { 
      clusters_.push_back( r ); 
      computeRawEnergy();
    }

    /// add reference to constituent BasicCluster
    void addPreshowerCluster( const CaloClusterPtr & r ) { preshowerClusters_.push_back( r ); }

    /** Set preshower planes status :
        0 : both planes working
        1 : only first plane working
        2 : only second plane working
        3 : both planes dead */        

    void setPreshowerPlanesStatus(const uint32_t& status){
      uint32_t flags = flags_ & flagsMask_;
      flags_= flags | (status << flagsOffset_);
    }

    /** Get preshower planes status :
        0 : both planes working
        1 : only first plane working
        2 : only second plane working
        3 : both planes dead */    
    const int  getPreshowerPlanesStatus() const {
      return (flags_>>flagsOffset_);
    }

  private:

    void computeRawEnergy();

    /// reference to BasicCluster seed
    CaloClusterPtr seed_;

    /// references to BasicCluster constitunets
    CaloClusterPtrVector clusters_;

    /// references to BasicCluster constitunets
    CaloClusterPtrVector preshowerClusters_;

    /// used hits by detId - retrieved from BC constituents -- now inherited from CaloCluster
    //std::vector<DetId> usedHits_;

    double preshowerEnergy_;

    double rawEnergy_;
    
    double phiWidth_;
    double etaWidth_;

    double preshowerEnergy1_;
    double preshowerEnergy2_;
  };

}
#endif
