#ifndef CALORECHIT_H
#define CALORECHIT_H 1

#include "DataFormats/DetId/interface/DetId.h"
#include <ostream>

namespace cms {

  /** \class CaloRecHit
    
  $Date: $
  $Revision: $
  \author J. Mans - Minnesota
  */
  class CaloRecHit {
  public:
    CaloRecHit(); // for persistence
    explicit CaloRecHit(float energy, float time);
    virtual ~CaloRecHit();
    float energy() const { return energy_; }
    float time() const { return time_; }
    virtual DetId genericId() const = 0;
  private:
    float energy_;
    float time_;
  };

  std::ostream& operator<<(std::ostream& s, const CaloRecHit& hit);
}
  
#endif
