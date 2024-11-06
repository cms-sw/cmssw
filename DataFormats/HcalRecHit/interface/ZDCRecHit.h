#ifndef DATAFORMATS_HCALRECHIT_ZDCRECHIT_H
#define DATAFORMATS_HCALRECHIT_ZDCRECHIT_H 1

#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"

/** \class ZDCRecHit
 *  
 *\author J. Mans - Minnesota
 */
class ZDCRecHit : public CaloRecHit {
public:
  typedef HcalZDCDetId key_type;

  ZDCRecHit();
  ZDCRecHit(const HcalZDCDetId& id, float energy, float time, float lowGainEnergy);
  /// get the id
  HcalZDCDetId id() const { return HcalZDCDetId(detid()); }
  // follow EcalRecHit method of adding variable flagBits_ to CaloRecHit
  float lowGainEnergy() const { return lowGainEnergy_; };

  constexpr inline void setEnergySOIp1(const float en) { energySOIp1_ = en; };
  constexpr inline float energySOIp1() const { return energySOIp1_; };  // energy of Slice of Interest plus 1
  constexpr inline void setRatioSOIp1(const float ratio) { ratioSOIp1_ = ratio; };
  constexpr inline float ratioSOIp1() const {
    return ratioSOIp1_;
  };  // ratio of Energy of (Slice of Interest)/ (Slice of Interest plus 1)
  constexpr inline void setTDCtime(const float time) { TDCtime_ = time; };
  constexpr inline float TDCtime() const { return TDCtime_; };
  constexpr inline void setChargeWeightedTime(const float time) {
    chargeWeightedTime_ = time;
  };  // time of activity determined by charged weighted average
  constexpr inline float chargeWeightedTime() const { return chargeWeightedTime_; };

private:
  float lowGainEnergy_;
  float energySOIp1_;
  float ratioSOIp1_;
  float TDCtime_;
  float chargeWeightedTime_;
};

std::ostream& operator<<(std::ostream& s, const ZDCRecHit& hit);

#endif
