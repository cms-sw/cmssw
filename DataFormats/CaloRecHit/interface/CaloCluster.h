#ifndef DataFormats_CaloRecHit_CaloCluster_h
#define DataFormats_CaloRecHit_CaloCluster_h

/** \class reco::CaloCluster 
 *  
 * Base class for all types calorimeter clusters
 *
 * \author Shahram Rahatlou, INFN
 *
 * Comments:
 * modified AlgoId enumeration to include cleaning status flags
 * In summary:
 * algoID_ < 200 object is in clean collection
 * algoID_ >=100 object is in unclean collection
 *
 */
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/CaloRecHit/interface/CaloID.h"

#include "DataFormats/DetId/interface/DetId.h"

#include <vector>
#include <string>
#include <iostream>

namespace reco {


  class CaloCluster {
  public:
    
    enum AlgoId { island = 0, hybrid = 1, fixedMatrix = 2, dynamicHybrid = 3, multi5x5 = 4, particleFlow = 5,  undefined = 1000};

    // super-cluster flags
    enum SCFlags { cleanOnly = 0, common = 100, uncleanOnly = 200 };

   //FIXME:  
   //temporary fix... to be removed before 310 final
   typedef AlgoId AlgoID ;
 
   /// default constructor. Sets energy and position to zero
    CaloCluster() : 
      energy_(0), correctedEnergy_(-1.0), correctedEnergyUncertainty_(-1.0), 
      algoID_( undefined ), flags_(0) {}

    /// constructor with algoId, to be used in all child classes
    CaloCluster(AlgoID algoID) : 
      energy_(0), correctedEnergy_(-1.0), correctedEnergyUncertainty_(-1.0),
      algoID_( algoID ), flags_(0) {}

    CaloCluster( double energy,
                 const math::XYZPoint& position,
                 const CaloID& caloID) :
      energy_ (energy), correctedEnergy_(-1.0), correctedEnergyUncertainty_(-1.0), position_ (position), caloID_(caloID),algoID_( undefined ), flags_(0) {}


    /// resets the CaloCluster (position, energy, hitsAndFractions)
    void reset();
    
     /// constructor from values 
     CaloCluster( double energy,  
 		 const math::XYZPoint& position ) : 
       energy_ (energy), correctedEnergy_(-1.0), position_ (position),algoID_( undefined ), flags_(0) {} 


    CaloCluster( double energy,
		 const math::XYZPoint& position,
		 const CaloID& caloID,
                 const AlgoID& algoID,
                 uint32_t flags = 0) :
      energy_ (energy), correctedEnergy_(-1.0), position_ (position), 
      caloID_(caloID), algoID_(algoID) {
      flags_=flags&flagsMask_;
    }

    CaloCluster( double energy,
                 const math::XYZPoint& position,
                 const CaloID& caloID,
                 const std::vector< std::pair< DetId, float > > &usedHitsAndFractions,
                 const AlgoId algoId,
		 const DetId seedId = DetId(0),
                 uint32_t flags = 0) :
      energy_ (energy), correctedEnergy_(-1.0), position_ (position), caloID_(caloID), 
      hitsAndFractions_(usedHitsAndFractions), algoID_(algoId),seedId_(seedId){
      flags_=flags&flagsMask_;
    }

   //FIXME:
   /// temporary compatibility constructor
    CaloCluster( double energy,
                 const math::XYZPoint& position,
                 float chi2,
                 const std::vector<DetId > &usedHits,
                 const AlgoId algoId,
                 uint32_t flags = 0) :
      energy_(energy), correctedEnergy_(-1.0), position_ (position),  algoID_(algoId)
       {
          hitsAndFractions_.reserve(usedHits.size());
          for(size_t i = 0; i < usedHits.size(); i++) hitsAndFractions_.push_back(std::pair< DetId, float > ( usedHits[i],1.));
	  flags_=flags&flagsMask_;
      }


    /// destructor
    virtual ~CaloCluster() {}


    void setEnergy(double energy){energy_ = energy;}
    void setCorrectedEnergy(double cenergy){correctedEnergy_ = cenergy;}
    void setCorrectedEnergyUncertainty(float energyerr) { correctedEnergyUncertainty_ = energyerr; }
    
    void setPosition(const math::XYZPoint& p){position_ = p;}

    void setCaloId(const CaloID& id) {caloID_= id;}

    void setAlgoId(const AlgoId& id) {algoID_=id;}

    void setSeed(const DetId& id) {seedId_=id;}

    /// cluster energy
    double energy() const { return energy_; }
    double correctedEnergy() const { return correctedEnergy_; }
    float correctedEnergyUncertainty() const { return correctedEnergyUncertainty_; }

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

    uint32_t flags() const { return flags_&flagsMask_; }
    void setFlags( uint32_t flags) { 
      uint32_t reserved = (flags_ & ~flagsMask_);
      flags_ = (reserved ) | (flags & flagsMask_); 
    }
    bool isInClean()   const { return flags() < uncleanOnly; }
    bool isInUnclean() const { return flags() >= common; }

    const CaloID& caloID() const {return caloID_;}

    void addHitAndFraction( DetId id, float fraction ) { 
            hitsAndFractions_.push_back( std::pair<DetId, float>(id, fraction) );
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

    /// return DetId of seed
    DetId seed() const { return seedId_; }

  protected:

    /// cluster energy
    double              energy_;
    double              correctedEnergy_;
    float               correctedEnergyUncertainty_;

    /// cluster centroid position
    math::XYZPoint      position_;

    /// bitmask for detector information
    CaloID              caloID_;
    
    // used hits by detId
    std::vector< std::pair<DetId, float> > hitsAndFractions_;

    // cluster algorithm Id
    AlgoID              algoID_;

    /// DetId of seed
    DetId		seedId_;

    /// flags (e.g. for handling of cleaned/uncleaned SC)
    /// 4  most significant bits  reserved
    /// 28 bits for handling of cleaned/uncleaned
    uint32_t            flags_;

    static const uint32_t flagsMask_  =0x0FFFFFFF;
    static const uint32_t flagsOffset_=28;
  };

}

#endif
