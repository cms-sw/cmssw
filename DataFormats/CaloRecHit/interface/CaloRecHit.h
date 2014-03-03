#ifndef DATAFORMATS_CALORECHIT_CALORECHIT_H
#define DATAFORMATS_CALORECHIT_CALORECHIT_H 1

#include "DataFormats/DetId/interface/DetId.h"
#include <ostream>


/** \class CaloRecHit
 * 
 * $Date: 2012/06/07 07:57:14 $
 * $Revision: 1.9 $
 *\author J. Mans - Minnesota
 */
class CaloRecHit {
public:
  CaloRecHit(); // for persistence
  explicit CaloRecHit(const DetId& id, float energy, float time, uint32_t flags = 0, uint32_t aux=0);

  float energy() const { return energy_; }
  void setEnergy(float energy) { energy_=energy; }
  float time() const { return time_; }
  void setTime(float time) { time_=time; }
  const DetId& detid() const { return id_; }
  uint32_t flags() const { return flags_; }
  void setFlags(uint32_t flags) { flags_=flags; }
  void setFlagField(uint32_t value, int base, int width=1);
  uint32_t flagField(int base, int width=1) const;
  void setAux(uint32_t value) { aux_=value; }
  uint32_t aux() const { return aux_; }
private:
  DetId id_;
  float energy_;
  float time_;
  uint32_t flags_;
  uint32_t aux_;
};

std::ostream& operator<<(std::ostream& s, const CaloRecHit& hit);
  
#endif
