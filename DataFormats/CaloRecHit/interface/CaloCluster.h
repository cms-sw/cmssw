#ifndef DataFormats_CaloRecHit_CaloCluster_h
#define DataFormats_CaloRecHit_CaloCluster_h

/** \class reco::CaloCluster 
 *  
 * Base class for all types calorimeter clusters
 *
 * \author Shahram Rahatlou, INFN
 *
 * \version $Id: CaloCluster.h,v 1.3 2008/04/25 08:43:25 cbern Exp $
 *
 */
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/CaloRecHit/interface/CaloID.h"

namespace reco {

  class CaloCluster {
    
  public:

    /// default constructor. Sets energy and position to zero
    CaloCluster() : energy_(0.) { }

    /// constructor from values
    CaloCluster( double energy, 
		 const math::XYZPoint& position ) :
      energy_ (energy), position_ (position) {}

    /// constructor from values
    CaloCluster( double energy, 
		 const math::XYZPoint& position, 
		 const CaloID& caloID) :
      energy_ (energy), position_ (position), caloID_(caloID) {}

    /// destructor
    virtual ~CaloCluster() {}

    /// cluster energy
    double energy() const { return energy_; }

    /// cluster centroid position
    const math::XYZPoint & position() const { return position_; }

    /// comparison >= operator
    bool operator >=(const CaloCluster& rhs) const { 
      return (energy_>=rhs.energy_); 
    }

    /// comparison > operator
    bool operator > (const CaloCluster& rhs) const { 
      return (energy_> rhs.energy_); 
    }

    /// comparison <= operator
    bool operator <=(const CaloCluster& rhs) const { 
      return (energy_<=rhs.energy_); 
    }

    /// comparison <= operator
    bool operator < (const CaloCluster& rhs) const { 
      return (energy_< rhs.energy_); 
    }

    /// x coordinate of cluster centroid
    double x() const { return position_.x(); }

    /// y coordinate of cluster centroid
    double y() const { return position_.y(); }

    /// z coordinate of cluster centroid
    double z() const { return position_.z(); }

    /// pseudorapidity of cluster centroid
    double eta() const { return position_.eta(); }

    /// azimuthal angle of cluster centroid
    double phi() const { return position_.phi(); }

    
    CaloID& caloID() {return caloID_;}
    const CaloID& caloID() const {return caloID_;}
    
    CaloCluster& operator=(const CaloCluster & rhs) {
      energy_ = rhs.energy_;
      position_ = rhs.position_;
      caloID_ = rhs.caloID_;
      return *this;
    }

    
    
  protected:

    /// cluster energy
    double              energy_;

    /// cluster centroid position
    math::XYZPoint      position_;

    /// bitmask for detector information
    CaloID              caloID_;

  };

}

#endif
