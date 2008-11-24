#ifndef CastorReco_CastorJet_h
#define CastorReco_CastorJet_h
/** \class reco::CastorJet CastorJet.h DataFormats/CastorReco/CastorJet.h
 *  
 * Class for Castor jets
 *
 * \author Hans Van Haevermaet, University of Antwerp
 *
 * \version $Id: CastorJet.h,v 1.1.2.1 2008/08/29 14:29:10 hvanhaev Exp $
 *
 */
#include <vector>
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/CastorReco/interface/CastorTower.h"
   
namespace reco {

  class CastorJet {
  public:

    /// default constructor. Sets energy and position to zero
    CastorJet() : energy_(0.), position_(ROOT::Math::XYZPoint(0.,0.,0.)), emEnergy_(0.), hadEnergy_(0.), emtotRatio_(0.), width_(0.),
    depth_(0.), usedTowers_(0) { }

    /// constructor from values
    CastorJet(const double energy, const ROOT::Math::XYZPoint& position, const double emEnergy, const double hadEnergy, const double emtotRatio, const double width,
    const double depth, const std::vector<CastorTower> usedTowers);

    /// destructor
    virtual ~CastorJet();

    /// jet energy
    double energy() const { return energy_; }

    /// jet centroid position
    ROOT::Math::XYZPoint position() const { return position_; }
    
    /// jet em energy
    double emEnergy() const { return emEnergy_; }
    
    /// jet had energy
    double hadEnergy() const { return hadEnergy_; }
    
    /// jet em/tot ratio
    double emtotRatio() const { return emtotRatio_; }
    
    /// jet width in phi
    double width() const { return width_; }
    
    /// jet depth in z
    double depth() const { return depth_; }
    
    /// get used CastorTowers
    std::vector<CastorTower> getUsedTowers() const { return usedTowers_; }

    /// comparison >= operator
    bool operator >=(const CastorJet& rhs) const { return (energy_>=rhs.energy_); }

    /// comparison > operator
    bool operator > (const CastorJet& rhs) const { return (energy_> rhs.energy_); }

    /// comparison <= operator
    bool operator <=(const CastorJet& rhs) const { return (energy_<=rhs.energy_); }

    /// comparison <= operator
    bool operator < (const CastorJet& rhs) const { return (energy_< rhs.energy_); }

    /// pseudorapidity of jet centroid
    double eta() const { return position_.eta(); }
    /// azimuthal angle of jet centroid
    double phi() const { return position_.phi(); }
  private:

    /// jet energy
    double energy_;
    
    /// jet centroid position
    ROOT::Math::XYZPoint position_;
    
    /// jet em energy
    double emEnergy_;
    
    /// jet had energy
    double hadEnergy_;
    
    /// jet em/tot Ratio
    double emtotRatio_;
    
    /// jet width
    double width_;
    
    /// jet depth
    double depth_;
    
    /// used CastorTowers
    std::vector<CastorTower> usedTowers_;

  };
  
  // define CastorJetCollection
  typedef std::vector<CastorJet> CastorJetCollection;

}

#endif
