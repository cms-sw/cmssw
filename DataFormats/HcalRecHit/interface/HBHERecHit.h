#ifndef DATAFORMATS_HCALRECHIT_HBHERECHIT_H
#define DATAFORMATS_HCALRECHIT_HBHERECHIT_H 1

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"


/** \class HBHERecHit
 *  
 * $Date: 2013/03/27 17:57:44 $
 * $Revision: 1.6 $
 * \author J. Mans - Minnesota
 */
class HBHERecHit : public CaloRecHit {
public:
  typedef HcalDetId key_type;

  HBHERecHit();
  //HBHERecHit(const HcalDetId& id, float energy, float time);
  /// get the id
  HBHERecHit(const HcalDetId& id, float amplitude, float timeRising, float timeFalling=0);
  /// get the amplitude (generally fC, but can vary)
  /// get the hit time
  float timeFalling() const { return timeFalling_; }
  HcalDetId id() const { return HcalDetId(detid()); }



private:

  float timeFalling_;
};

std::ostream& operator<<(std::ostream& s, const HBHERecHit& hit);


#endif
