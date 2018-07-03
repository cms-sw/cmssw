#ifndef DATAFORMATS_HCALRECHIT_HFRECHIT_H
#define DATAFORMATS_HCALRECHIT_HFRECHIT_H 1

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"


/** \class HFRecHit
 *  
 *\author J. Mans - Minnesota
 */
class HFRecHit : public CaloRecHit {
public:
  typedef HcalDetId key_type;

  constexpr HFRecHit() 
    : CaloRecHit(), timeFalling_(0.f), auxHF_(0)
  {}
  
  //HFRecHit(const HcalDetId& id, float energy, float time);
  /// get the id
  constexpr HFRecHit(const HcalDetId& id, float en, float timeRising, 
                     float timeFalling=0) 
    : CaloRecHit(id,en,timeRising), timeFalling_(timeFalling), auxHF_(0)
  {}

  /// get the amplitude (generally fC, but can vary)
  /// get the hit time
  constexpr float timeFalling() const { return timeFalling_; }
  constexpr void setTimeFalling(float timeFalling) { timeFalling_ = timeFalling; }
  constexpr HcalDetId id() const { return HcalDetId(detid()); }

  constexpr void setAuxHF(const uint32_t u) {auxHF_ = u;}
  constexpr uint32_t getAuxHF() const {return auxHF_;}

private:

  float timeFalling_;
  uint32_t auxHF_;
};

std::ostream& operator<<(std::ostream& s, const HFRecHit& hit);

#endif
