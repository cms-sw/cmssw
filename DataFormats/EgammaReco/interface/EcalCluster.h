#ifndef DataFormats_EgammaReco_h
#define DataFormats_EgammaReco_h

#include <vector>
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/DetId/interface/DetId.h"

namespace reco {

class EcalCluster {

 public:
   EcalCluster() : energy_(0.), position_(math::XYZPoint(0.,0.,0.)) {}
   EcalCluster(const double energy, const math::XYZPoint& position);

   // accessors
   double energy() const { return energy_; }
   math::XYZPoint position() const { return position_; }

   // operators
   bool operator >=(const EcalCluster& rhs) const { return (energy_>=rhs.energy_); }
   bool operator > (const EcalCluster& rhs) const { return (energy_> rhs.energy_); }
   bool operator <=(const EcalCluster& rhs) const { return (energy_<=rhs.energy_); }
   bool operator < (const EcalCluster& rhs) const { return (energy_< rhs.energy_); }

  // bool operator == // must use ids!


  // virtual methods
  virtual ~EcalCluster();
  virtual std::vector<DetId> getHitsByDetId() const = 0; // vector of used hits
                                                         // must be implemented by all
                                                         // derived instances

 private:
   double              energy_; // cluster energy
   math::XYZPoint   position_; // cluster position

}; // EcalCluster


} // namespace
#endif
