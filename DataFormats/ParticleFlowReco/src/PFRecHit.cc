#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include <limits>
namespace reco {

  void PFRecHit::addNeighbour(short x, short y, short z, unsigned int ref) {
    //bitmask interface  to accomodate more advanced naighbour finding [i.e in z as well]
    //bit 0 side for eta [0 for <=0 , 1 for >0]
    //bits 1,2,3 : abs(eta) wrt the center
    //bit 4 side for phi
    //bits 5,6,7 : abs(phi) wrt the center
    //bit 8 side for z
    //bits 9,10,11 : abs(z) wrt the center

    unsigned short absx = std::abs(x);
    unsigned short absy = std::abs(y);
    unsigned short absz = std::abs(z);

    unsigned short bitmask = 0;

    if (x > 0)
      bitmask = bitmask | 1;
    bitmask = bitmask | (absx << 1);
    if (y > 0)
      bitmask = bitmask | (1 << 4);
    bitmask = bitmask | (absy << 5);
    if (z > 0)
      bitmask = bitmask | (1 << 8);
    bitmask = bitmask | (absz << 9);

    auto pos = neighbours_.size();
    if (z == 0) {
      pos = neighbours8_++;
      //find only the 4 neighbours
      if (absx + absy == 1)
        pos = neighbours4_++;
    }
    neighbours_.insert(neighbours_.begin() + pos, ref);
    neighbourInfos_.insert(neighbourInfos_.begin() + pos, bitmask);

    assert(neighbours4_ < 5);
    assert(neighbours8_ < 9);
    assert(neighbours4_ <= neighbours8_);
    assert(neighbours8_ <= neighbours_.size());
  }

  unsigned int PFRecHit::getNeighbour(short x, short y, short z) const {
    unsigned short absx = abs(x);
    unsigned short absy = abs(y);
    unsigned short absz = abs(z);

    unsigned short bitmask = 0;

    if (x > 0)
      bitmask = bitmask | 1;
    bitmask = bitmask | (absx << 1);
    if (y > 0)
      bitmask = bitmask | (1 << 4);
    bitmask = bitmask | (absy << 5);
    if (z > 0)
      bitmask = bitmask | (1 << 8);
    bitmask = bitmask | (absz << 9);

    for (unsigned int i = 0; i < neighbourInfos_.size(); ++i) {
      if (neighbourInfos_[i] == bitmask)
        return neighbours_[i];
    }
    return std::numeric_limits<unsigned int>::max();
  }

}  // namespace reco

std::ostream& operator<<(std::ostream& out, const reco::PFRecHit& hit) {
  if (!out)
    return out;

  out << "hit id:" << hit.detId() << " l:" << hit.layer() << " E:" << hit.energy() << " t:" << hit.time();
  if (hit.hasCaloCell()) {
    auto const& pos = hit.positionREP();
    out << " rep:" << pos.rho() << "," << pos.eta() << "," << pos.phi() << "|";
  }
  return out;
}
