#ifndef CastorReco_CastorTower_h
#define CastorReco_CastorTower_h
/** \class reco::CastorTower CastorTower.h DataFormats/CastorReco/CastorTower.h
 *  
 * Class for Castor towers
 *
 * \author Hans Van Haevermaet, University of Antwerp
 *
 * \version $Id: CastorTower.h,v 1.6 2012/10/14 13:39:11 innocent Exp $
 *
 */
#include <vector>
#include <memory>
#include "DataFormats/Math/interface/Point3D.h"

#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/SortedCollection.h"

#include "DataFormats/HcalRecHit/interface/CastorRecHit.h"
//#include "DataFormats/HcalRecHit/interface/HcalRecHitFwd.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"

namespace reco {

  class CastorTower : public LeafCandidate {
  public:
     
    typedef edm::SortedCollection<CastorRecHit> CastorRecHitCollection;
    typedef edm::Ref<CastorRecHitCollection> CastorRecHitRef;
    typedef edm::RefVector<CastorRecHitCollection> CastorRecHitRefs;
    
    // default constructor. Set energy and position to zero
 
    CastorTower() : energy_(0.), position_(ROOT::Math::XYZPoint(0.,0.,0.)), emEnergy_(0.), hadEnergy_(0.), fem_(0.), depth_(0.), fhot_(0.) { }

    // constructor from values
    CastorTower(const double energy, const ROOT::Math::XYZPoint& position, const double emEnergy, const double hadEnergy, const double fem,
		const double depth, const double fhot, const CastorRecHitRefs& usedRecHits);

    /// destructor
    virtual ~CastorTower();

    /// tower centroid position
    ROOT::Math::XYZPoint position() const { return position_; }
    
    /// tower em energy
    double emEnergy() const { return emEnergy_; }
    
    /// tower had energy
    double hadEnergy() const { return hadEnergy_; }
    
    /// tower em/tot ratio
    double fem() const { return fem_; }
    
    /// tower depth in z
    double depth() const { return depth_; }
    
    /// tower  hotcell/tot ratio
    double fhot() const { return fhot_; }

    /// vector of used RecHits
    CastorRecHitRefs getUsedRecHits() const { return usedRecHits_; }

    /// fist iterator over CastorRecHit constituents
    CastorRecHitRefs::iterator rechitsBegin() const { return usedRecHits_.begin(); }

    /// last iterator over CastorRecHit constituents
    CastorRecHitRefs::iterator rechitsEnd() const { return usedRecHits_.end(); }

    /// number of CastorRecHit constituents
    size_t rechitsSize() const { return usedRecHits_.size(); }

    /// add reference to constituent CastorRecHit
    void add( const CastorRecHitRef& rechit ) { usedRecHits_.push_back( rechit ); }

    /// comparison >= operator
    bool operator >=(const CastorTower& rhs) const { return (energy_>=rhs.energy_); }

    /// comparison > operator
    bool operator > (const CastorTower& rhs) const { return (energy_> rhs.energy_); }

    /// comparison <= operator
    bool operator <=(const CastorTower& rhs) const { return (energy_<=rhs.energy_); }

    /// comparison <= operator
    bool operator < (const CastorTower& rhs) const { return (energy_< rhs.energy_); }

    /// x of tower centroid
    double xPos() const { return position_.x(); }

    /// y of tower centroid
    double yPos() const { return position_.y(); }

    /// rho of tower centroid
    double rho() const { return position_.rho(); }

  private:

    /// tower energy
    double energy_;
    
    /// tower centroid position
    ROOT::Math::XYZPoint position_;
    
    /// tower em energy
    double emEnergy_;
    
    /// tower had energy
    double hadEnergy_;
    
    /// tower em/tot Ratio
    double fem_;
    
    /// tower depth
    double depth_;

    /// tower  hotcell/tot ratio
    double fhot_;

    /// references to CastorRecHit constituents
    CastorRecHitRefs usedRecHits_;
  };
  
  /// collection of CastorTower objects
  typedef std::vector<CastorTower> CastorTowerCollection;

  // persistent reference to CastorTower objects
  typedef edm::Ref<CastorTowerCollection> CastorTowerRef;

  /// vector of references to CastorTower objects all in the same collection
  typedef edm::RefVector<CastorTowerCollection> CastorTowerRefVector;

  /// iterator over a vector of references to CastorTower objects all in the same collection
  typedef CastorTowerRefVector::iterator CastorTower_iterator;

}

#endif
