#ifndef DataFormats_CaloRecHit_CaloCluster_h
#define DataFormats_CaloRecHit_CaloCluster_h

/** \class reco::CaloCluster 
 *  
 * Base class for all types calorimeter clusters
 *
 * \author Shahram Rahatlou, INFN
 *
 * \version $Id: CaloCluster.h,v 1.11 2009/02/09 12:14:16 cbern Exp $
 *
 */
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/CaloRecHit/interface/CaloID.h"

#include "DataFormats/DetId/interface/DetId.h"

#include <vector>
#include <string>
#include <iostream>

namespace reco {

  //FIXME: to be moved inside CaloCluster
  enum AlgoId { island = 0, hybrid = 1, fixedMatrix = 2, dynamicHybrid = 3, multi5x5 = 4, pFClusters = 5 , ALGO_undefined = 100};

  class CaloCluster {
  public:

   //FIXME:  
   //temporary fix... to be removed before 310 final
   typedef AlgoId AlgoID ;
 
   /// default constructor. Sets energy and position to zero
    CaloCluster() : 
      energy_(0), 
      algoID_( ALGO_undefined ) {}

    /// constructor with algoId, to be used in all child classes
    CaloCluster(AlgoID algoID) : 
      energy_(0), 
      algoID_( algoID ) {}

    CaloCluster( double energy,
                 const math::XYZPoint& position,
                 const CaloID& caloID) :
      energy_ (energy), position_ (position), caloID_(caloID) {}


    /// resets the CaloCluster (position, energy, hitsAndFractions)
    void reset();
    
     /// constructor from values 
     CaloCluster( double energy,  
 		 const math::XYZPoint& position ) : 
       energy_ (energy), position_ (position) {} 


    CaloCluster( double energy,
		 const math::XYZPoint& position,
		 const CaloID& caloID,
                 const AlgoID& algoID) :
      energy_ (energy), position_ (position), 
      caloID_(caloID), algoID_(algoID) {}

    CaloCluster( double energy,
                 const math::XYZPoint& position,
                 const CaloID& caloID,
                 const std::vector< std::pair< DetId, float > > &usedHitsAndFractions,
                 const AlgoId algoId) :
      energy_ (energy), position_ (position), caloID_(caloID), hitsAndFractions_(usedHitsAndFractions), algoID_(algoId) {}

   //FIXME:
   /// temporary compatibility constructor
    CaloCluster( double energy,
                 const math::XYZPoint& position,
                 float chi2,
                 const std::vector<DetId > &usedHits,
                 const AlgoId algoId) :
      energy_ (energy), position_ (position),  algoID_(algoId)
       {
          hitsAndFractions_.reserve(usedHits.size());
          for(size_t i = 0; i < usedHits.size(); i++) hitsAndFractions_.push_back(std::pair< DetId, float > ( usedHits[i],1.));
      }


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
    AlgoId algo() const { return algoID_; }
    AlgoID algoID() const { return algo(); }
    
    const CaloID& caloID() const {return caloID_;}

    void addHitAndFraction( DetId id, float fraction ) { 
            hitsAndFractions_.push_back( std::pair<DetId, float>(id, fraction) );
    }

    CaloCluster& operator=(const CaloCluster & rhs) {
      energy_ = rhs.energy_;
      position_ = rhs.position_;
      caloID_ = rhs.caloID_;
      hitsAndFractions_ = rhs.hitsAndFractions_;
      algoID_ = rhs.algoID_;
      return *this;
    }

    /// replace getHitsByDetId() : return hits by DetId 
    /// and their corresponding fraction of energy considered
    /// to compute the total cluster energy
    const std::vector< std::pair<DetId, float> > & hitsAndFractions() const { return hitsAndFractions_; }
    
    /// print hitAndFraction
    std::string printHitAndFraction(unsigned i) const;

    /// print me
    friend std::ostream& operator<<(std::ostream& out, 
				    const CaloCluster& cluster);

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
    AlgoID              algoID_;
  };

}

#endif
