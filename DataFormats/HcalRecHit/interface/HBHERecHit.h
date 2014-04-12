#ifndef DATAFORMATS_HCALRECHIT_HBHERECHIT_H
#define DATAFORMATS_HCALRECHIT_HBHERECHIT_H 1

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"


/** \class HBHERecHit
 *  
 * \author J. Mans - Minnesota
 */
class HBHERecHit : public CaloRecHit {
public:
  typedef HcalDetId key_type;

  HBHERecHit();
  //HBHERecHit(const HcalDetId& id, float energy, float time);
  HBHERecHit(const HcalDetId& id, float amplitude, float timeRising, float timeFalling=0);

  /// get the hit falling time
  float timeFalling() const { return timeFalling_; }
  /// get the id
  HcalDetId id() const { return HcalDetId(detid()); }

  inline void setRawEnergy(const float en) {rawEnergy_ = en;}
  inline float eraw() const {return rawEnergy_;}

private:
  float timeFalling_;
  float rawEnergy_;
};

std::ostream& operator<<(std::ostream& s, const HBHERecHit& hit);


#endif
