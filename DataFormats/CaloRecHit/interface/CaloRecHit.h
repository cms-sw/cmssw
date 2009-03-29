#ifndef DATAFORMATS_CALORECHIT_CALORECHIT_H
#define DATAFORMATS_CALORECHIT_CALORECHIT_H 1

#include "DataFormats/DetId/interface/DetId.h"
#include <ostream>


/** \class CaloRecHit
 * 
 * $Date: 2008/11/08 10:30:10 $
 * $Revision: 1.5 $
 *\author J. Mans - Minnesota
 */
class CaloRecHit {
public:
  CaloRecHit(); // for persistence
  explicit CaloRecHit(const DetId& id, float energy, float time, uint32_t flags = 0);
  virtual ~CaloRecHit();
  float energy() const { return energy_; }
  float time() const { return time_; }
  const DetId& detid() const { return id_; }
  uint32_t flags() const { return flags_; }
  void setFlags(uint32_t flags) { flags_=flags; }
  void setFlagField(uint32_t value, int base, int width=1);
  uint32_t flagField(int base, int width=1) const;
private:
  DetId id_;
  float energy_;
  float time_;
  uint32_t flags_;
};

std::ostream& operator<<(std::ostream& s, const CaloRecHit& hit);
  
#endif
