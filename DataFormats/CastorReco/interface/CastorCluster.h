#ifndef CastorReco_CastorCluster_h
#define CastorReco_CastorCluster_h
/** \class reco::CastorCluster CastorCluster.h DataFormats/CastorReco/CastorCluster.h
 *  
 * Class for Castor clusters
 *
 * \author Hans Van Haevermaet, University of Antwerp
 *
 * \version $Id: CastorCluster.h,v 1.1 2009/02/27 15:48:37 hvanhaev Exp $
 *
 */

#include <vector>
#include <memory>
#include "DataFormats/Math/interface/Point3D.h"

#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "DataFormats/CastorReco/interface/CastorTower.h"

namespace reco {

  class CastorCluster {
  public:

    /// default constructor. Sets energy and position to zero
    CastorCluster() : energy_(0.), position_(ROOT::Math::XYZPoint(0.,0.,0.)), emEnergy_(0.), hadEnergy_(0.), fem_(0.), width_(0.),
    depth_(0.), fhot_(0.), sigmaz_(0.) { }

    /// constructor from values
    CastorCluster(const double energy, const ROOT::Math::XYZPoint& position, const double emEnergy, const double hadEnergy, const double fem, 
		  const double width, const double depth, const double fhot, const double sigmaz, const CastorTowerRefVector& usedTowers);


    /// destructor
    virtual ~CastorCluster();

    /// cluster energy
    double energy() const { return energy_; }

    /// cluster centroid position
    ROOT::Math::XYZPoint position() const { return position_; }
    
    /// cluster em energy
    double emEnergy() const { return emEnergy_; }
    
    /// cluster had energy
    double hadEnergy() const { return hadEnergy_; }
    
    /// cluster em/tot ratio
    double fem() const { return fem_; }
    
    /// cluster width in phi
    double width() const { return width_; }
    
    /// cluster depth in z
    double depth() const { return depth_; }
    
    /// cluster hotcell/tot ratio
    double fhot() const { return fhot_; }

    /// cluster sigma z
    double sigmaz() const { return sigmaz_; }

    /// vector of used Towers
    CastorTowerRefVector getUsedTowers() const { return usedTowers_; }
    
    /// fist iterator over CastorTower constituents
    CastorTower_iterator towersBegin() const { return usedTowers_.begin(); }
    
    /// last iterator over CastorTower constituents
    CastorTower_iterator towersEnd() const { return usedTowers_.end(); }
    
    /// number of CastorTower constituents
    size_t towersSize() const { return usedTowers_.size(); }
    
    /// add reference to constituent CastorTower
    void add( const CastorTowerRef & tower ) { usedTowers_.push_back( tower ); }

    /// comparison >= operator
    bool operator >=(const CastorCluster& rhs) const { return (energy_>=rhs.energy_); }

    /// comparison > operator
    bool operator > (const CastorCluster& rhs) const { return (energy_> rhs.energy_); }

    /// comparison <= operator
    bool operator <=(const CastorCluster& rhs) const { return (energy_<=rhs.energy_); }

    /// comparison <= operator
    bool operator < (const CastorCluster& rhs) const { return (energy_< rhs.energy_); }

    /// pseudorapidity of cluster centroid
    double eta() const { return position_.eta(); }

    /// azimuthal angle of cluster centroid
    double phi() const { return position_.phi(); }

    /// x of cluster centroid
    double x() const { return position_.x(); }

    /// y of cluster centroid
    double y() const { return position_.y(); }

    /// rho of cluster centroid
    double rho() const { return position_.rho(); }

  private:

    /// cluster energy
    double energy_;

    /// cluster centroid position
    ROOT::Math::XYZPoint position_;
    
    /// cluster em energy
    double emEnergy_;
    
    /// cluster had energy
    double hadEnergy_;
    
    /// cluster em/tot Ratio
    double fem_;
    
    /// cluster width
    double width_;
    
    /// cluster depth
    double depth_;

    /// cluster hotcell/tot ratio
    double fhot_;

    /// cluster sigma z
    double sigmaz_;

    /// references to CastorTower constituents
    CastorTowerRefVector usedTowers_;
  };
  
  /// collection of CastorCluster objects
  typedef std::vector<CastorCluster> CastorClusterCollection;

  // persistent reference to CastorCluster objects
  typedef edm::Ref<CastorClusterCollection> CastorClusterRef;

  /// vector of references to CastorCluster objects all in the same collection
  typedef edm::RefVector<CastorClusterCollection> CastorClusterRefVector;

  /// iterator over a vector of references to CastorCluster objects all in the same collection
  typedef CastorClusterRefVector::iterator CastorCluster_iterator;
}

#endif
