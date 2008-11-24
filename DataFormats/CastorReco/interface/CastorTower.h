#ifndef CastorReco_CastorTower_h
#define CastorReco_CastorTower_h
/** \class reco::CastorTower CastorTower.h DataFormats/CastorReco/CastorTower.h
 *  
 * Class for Castor towers
 *
 * \author Hans Van Haevermaet, University of Antwerp
 *
 * \version $Id: CastorTower.h,v 1.1.2.1 2008/08/29 14:29:10 hvanhaev Exp $
 *
 */
#include <vector>
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/CastorReco/interface/CastorCell.h"
   
namespace reco {

  class CastorTower {
  public:

    /// default constructor. Sets energy and position to zero
    CastorTower() : energy_(0.), position_(ROOT::Math::XYZPoint(0.,0.,0.)), emEnergy_(0.), hadEnergy_(0.), emtotRatio_(0.),
    width_(0.), depth_(0.), usedCells_(0) { }

    /// constructor from values
    CastorTower(const double energy, const ROOT::Math::XYZPoint& position, const double emEnergy, const double hadEnergy, const double emtotRatio, const double width,
    const double depth, const std::vector<CastorCell> usedCells);

    /// destructor
    virtual ~CastorTower();

    /// tower energy
    double energy() const { return energy_; }

    /// tower centroid position
    ROOT::Math::XYZPoint position() const { return position_; }
    
    /// tower em energy
    double emEnergy() const { return emEnergy_; }
    
    /// tower had energy
    double hadEnergy() const { return hadEnergy_; }
    
    /// tower em/tot ratio
    double emtotRatio() const { return emtotRatio_; }
    
    /// tower width in phi
    double width() const { return width_; }
    
    /// tower depth in z
    double depth() const { return depth_; }
    
    /// vector of usedCells
    std::vector<CastorCell> getUsedCells() const { return usedCells_; }

    /// comparison >= operator
    bool operator >=(const CastorTower& rhs) const { return (energy_>=rhs.energy_); }

    /// comparison > operator
    bool operator > (const CastorTower& rhs) const { return (energy_> rhs.energy_); }

    /// comparison <= operator
    bool operator <=(const CastorTower& rhs) const { return (energy_<=rhs.energy_); }

    /// comparison <= operator
    bool operator < (const CastorTower& rhs) const { return (energy_< rhs.energy_); }

    /// pseudorapidity of tower centroid
    double eta() const { return position_.eta(); }
    /// azimuthal angle of tower centroid
    double phi() const { return position_.phi(); }
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
    double emtotRatio_;
    
    /// tower width
    double width_;
    
    /// tower depth
    double depth_;

    /// used CastorCells
    std::vector<CastorCell> usedCells_;

  };
  
  // define CastorTowerCollection
  typedef std::vector<CastorTower> CastorTowerCollection;

}

#endif
