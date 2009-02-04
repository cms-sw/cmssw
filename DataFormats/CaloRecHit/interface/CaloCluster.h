#ifndef DataFormats_CaloRecHit_CaloCluster_h
#define DataFormats_CaloRecHit_CaloCluster_h

/** \class reco::CaloCluster 
 *  
 * Base class for all types calorimeter clusters
 *
 * \author Shahram Rahatlou, INFN
 *
 * \version $Id: CaloCluster.h,v 1.6 2009/01/28 16:20:04 ferriff Exp $
 *
 */
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/CaloRecHit/interface/CaloID.h"

#include "DataFormats/DetId/interface/DetId.h"

#include <vector>

namespace reco {

  enum AlgoId { island = 0, hybrid = 1, fixedMatrix = 2, dynamicHybrid = 3, multi5x5 = 4, pFClusters = 5 };

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

    /// constructor from values
    CaloCluster( double energy, 
		 const math::XYZPoint& position, 
		 const CaloID& caloID,
                 const std::vector< std::pair< DetId, float > > &usedHitsAndFractions,
                 const AlgoId algoId) :
      energy_ (energy), position_ (position), caloID_(caloID), hitsAndFractions_(usedHitsAndFractions), algoId_(algoId) {}

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

    /// comparison < operator
    bool operator < (const CaloCluster& rhs) const { 
      return (energy_< rhs.energy_); 
    }

    /// comparison == operator
    bool operator==(const CaloCluster& rhs) const {
            return (energy_ == rhs.energy_);
    };

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

    /// size in number of hits (e.g. in crystals for ECAL)
    size_t size() const { return hitsAndFractions_.size(); }

    /// algorithm identifier
    AlgoId algo() const { return algoId_; }
    
    CaloID& caloID() {return caloID_;}
    const CaloID& caloID() const {return caloID_;}

    void addHitAndFraction( DetId id, float fraction ) { 
            hitsAndFractions_.push_back( std::pair<DetId, float>(id, fraction) );
    }

    CaloCluster& operator=(const CaloCluster & rhs) {
      energy_ = rhs.energy_;
      position_ = rhs.position_;
      caloID_ = rhs.caloID_;
      return *this;
    }

    /// replace getHitsByDetId() : return hits by DetId 
    /// and their corresponding fraction of energy considered
    /// to compute the total cluster energy
    const std::vector< std::pair<DetId, float> > & hitsAndFractions() const { return hitsAndFractions_; }
    
    
  protected:

    /// cluster energy
    double              energy_;

    /// cluster centroid position
    math::XYZPoint      position_;

    /// bitmask for detector information
    CaloID              caloID_;
    
    // used hits by detId
    std::vector< std::pair<DetId, float> > hitsAndFractions_;

    // cluster algorithm Id
    AlgoId              algoId_;
  };

}

#endif
