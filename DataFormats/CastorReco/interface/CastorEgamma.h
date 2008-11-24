#ifndef CastorReco_CastorEgamma_h
#define CastorReco_CastorEgamma_h
/** \class reco::CastorEgamma CastorEgamma.h DataFormats/CastorReco/CastorEgamma.h
 *  
 * Class for Castor electrons/photons
 *
 * \author Hans Van Haevermaet, University of Antwerp
 *
 * \version $Id: CastorEgamma.h,v 1.1.2.1 2008/08/29 14:29:10 hvanhaev Exp $
 *
 */
#include <vector>
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/CastorReco/interface/CastorJet.h"
#include "FWCore/Framework/interface/Event.h"

using namespace edm;

namespace reco {

  class CastorEgamma {
  public:

    /// default constructor. Sets energy and position to zero
    CastorEgamma() : energy_(0.), position_(ROOT::Math::XYZPoint(0.,0.,0.)), emEnergy_(0.), hadEnergy_(0.), emtotRatio_(0.), width_(0.),
    depth_(0.), usedJets_(0) { }

    /// constructor from values
    CastorEgamma(const double energy, const ROOT::Math::XYZPoint& position, const double emEnergy, const double hadEnergy, const double emtotRatio, const double width,
    const double depth, const std::vector<CastorJet> usedJets);

    /// destructor
    virtual ~CastorEgamma();

    /// Egamma energy
    double energy() const { return energy_; }

    /// Egamma centroid position
    ROOT::Math::XYZPoint position() const { return position_; }
    
    /// Egamma em energy
    double emEnergy() const { return emEnergy_; }
    
    /// Egamma had energy
    double hadEnergy() const { return hadEnergy_; }
    
    /// Egamma em/tot ratio
    double emtotRatio() const { return emtotRatio_; }
    
    /// Egamma width in phi
    double width() const { return width_; }
    
    /// Egamma depth in z
    double depth() const { return depth_; }
    
    /// Get used CastorJets
    std::vector<CastorJet> getUsedJets() const { return usedJets_; } 

    /// comparison >= operator
    bool operator >=(const CastorEgamma& rhs) const { return (energy_>=rhs.energy_); }

    /// comparison > operator
    bool operator > (const CastorEgamma& rhs) const { return (energy_> rhs.energy_); }

    /// comparison <= operator
    bool operator <=(const CastorEgamma& rhs) const { return (energy_<=rhs.energy_); }

    /// comparison <= operator
    bool operator < (const CastorEgamma& rhs) const { return (energy_< rhs.energy_); }

    /// pseudorapidity of Egamma centroid
    double eta() const { return position_.eta(); }
    /// azimuthal angle of Egamma centroid
    double phi() const { return position_.phi(); }
  private:

    /// Egamma energy
    double energy_;
    
    /// Egamma centroid position
    ROOT::Math::XYZPoint position_;
    
    /// Egamma em energy
    double emEnergy_;
    
    /// Egamma had energy
    double hadEnergy_;
    
    /// Egamma em/tot Ratio
    double emtotRatio_;
    
    /// Egamma width
    double width_;
    
    /// Egamma depth
    double depth_;
    
    /// used CastorJets
    std::vector<CastorJet> usedJets_;

  };
  
  // define CastorEgammaCollection
  typedef std::vector<CastorEgamma> CastorEgammaCollection;

}

#endif
