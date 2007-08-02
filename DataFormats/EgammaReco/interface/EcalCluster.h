#ifndef EgammaReco_EcalCluster_h
#define EgammaReco_EcalCluster_h
/** \class reco::EcalCluster EcalCluster.h DataFormats/EgammaReco/EcalCluster.h
 *  
 * Base class for all types of Ecal clusters
 *
 * \author Shahram Rahatlou, INFN
 *
 * \version $Id: EcalCluster.h,v 1.5 2006/06/18 17:36:53 llista Exp $
 *
 */
#include <vector>
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/DetId/interface/DetId.h"

namespace reco {

  class EcalCluster {
  public:

    /// default constructor. Sets energy and position to zero
    EcalCluster() : energy_(0.), position_(math::XYZPoint(0.,0.,0.)) { }

    /// constructor from values
    EcalCluster(const double energy, const math::XYZPoint& position);

    /// destructor
    virtual ~EcalCluster();

    /// cluster energy
    double energy() const { return energy_; }

    /// cluster centroid position
    const math::XYZPoint & position() const { return position_; }

    /// comparison >= operator
    bool operator >=(const EcalCluster& rhs) const { return (energy_>=rhs.energy_); }

    /// comparison > operator
    bool operator > (const EcalCluster& rhs) const { return (energy_> rhs.energy_); }

    /// comparison <= operator
    bool operator <=(const EcalCluster& rhs) const { return (energy_<=rhs.energy_); }

    /// comparison <= operator
    bool operator < (const EcalCluster& rhs) const { return (energy_< rhs.energy_); }

    /// vector of used hits
    /// Myst be implemented in  all derived classes
    virtual std::vector<DetId> getHitsByDetId() const = 0;
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
  private:

    /// cluster energy
    double              energy_;

    /// cluster centroid position
    math::XYZPoint   position_;
  };

}

#endif
