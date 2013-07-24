#ifndef DATAFORMATS_HCALRECHIT_HFRECHIT_H
#define DATAFORMATS_HCALRECHIT_HFRECHIT_H 1

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"


/** \class HFRecHit
 *  
 * $Date: 2013/03/27 17:57:44 $
 * $Revision: 1.6 $
 *\author J. Mans - Minnesota
 */
class HFRecHit : public CaloRecHit {
public:
  typedef HcalDetId key_type;

  HFRecHit();
  //HFRecHit(const HcalDetId& id, float energy, float time);
  /// get the id
  HFRecHit(const HcalDetId& id, float amplitude, float timeRising, float timeFalling=0);
  /// get the amplitude (generally fC, but can vary)
  /// get the hit time
  float timeFalling() const { return timeFalling_; }
  HcalDetId id() const { return HcalDetId(detid()); }

private:

  float timeFalling_;

};

std::ostream& operator<<(std::ostream& s, const HFRecHit& hit);

#endif
