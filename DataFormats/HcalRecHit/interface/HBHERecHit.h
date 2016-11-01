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
  inline float timeFalling() const { return timeFalling_; }
  /// get the id
  inline HcalDetId id() const { return HcalDetId(detid()); }

  inline void setChiSquared(const float chi2) {chiSquared_ = chi2;}
  inline float chi2() const {return chiSquared_;}

  inline void setRawEnergy(const float en) {rawEnergy_ = en;}
  inline float eraw() const {return rawEnergy_;}

  inline void setAuxEnergy(const float en) {auxEnergy_ = en;}
  inline float eaux() const {return auxEnergy_;}

  inline void setAuxHBHE(const uint32_t aux) { auxHBHE_ = aux;}
  inline uint32_t auxHBHE() const {return auxHBHE_;}

  inline void setAuxPhase1(const uint32_t aux) { auxPhase1_ = aux;}
  inline uint32_t auxPhase1() const {return auxPhase1_;}

private:
  float timeFalling_;
  float chiSquared_;
  float rawEnergy_;
  float auxEnergy_;
  uint32_t auxHBHE_;
  uint32_t auxPhase1_;
};

std::ostream& operator<<(std::ostream& s, const HBHERecHit& hit);


#endif
