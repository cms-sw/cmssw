#ifndef CALORECHIT_H
#define CALORECHIT_H 1

#include "DataFormats/DetId/interface/DetId.h"
#include <ostream>

namespace cms {

  /** \class CaloRecHit
    
  $Date: 2005/09/15 14:44:25 $
  $Revision: 1.2 $
  \author J. Mans - Minnesota
  */
  class CaloRecHit {
  public:
    CaloRecHit(); // for persistence
    explicit CaloRecHit(const DetId& id, float energy, float time);
    virtual ~CaloRecHit();
    float energy() const { return energy_; }
    float time() const { return time_; }
    const DetId& detid() const { return id_; }
  private:
    DetId id_;
    float energy_;
    float time_;
  };

  std::ostream& operator<<(std::ostream& s, const CaloRecHit& hit);
}
  
#endif
