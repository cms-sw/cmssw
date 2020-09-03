#ifndef DATAFORMATS_HCALRECHIT_HBHERECHIT_H
#define DATAFORMATS_HCALRECHIT_HBHERECHIT_H 1

#include <vector>

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"

/** \class HBHERecHit
 *  
 * \author J. Mans - Minnesota
 */
class HBHERecHit : public CaloRecHit {
public:
  typedef HcalDetId key_type;

  constexpr HBHERecHit()
      : CaloRecHit(),
        timeFalling_(0),
        chiSquared_(-1),
        rawEnergy_(-1.0e21),
        auxEnergy_(-1.0e21),
        auxHBHE_(0),
        auxPhase1_(0),
        auxTDC_(0) {}

  constexpr HBHERecHit(const HcalDetId& id, float energy, float timeRising, float timeFalling = 0)
      : CaloRecHit(id, energy, timeRising),
        timeFalling_(timeFalling),
        chiSquared_(-1),
        rawEnergy_(-1.0e21),
        auxEnergy_(-1.0e21),
        auxHBHE_(0),
        auxPhase1_(0),
        auxTDC_(0) {}

  /// get the hit falling time
  constexpr inline float timeFalling() const { return timeFalling_; }
  constexpr inline void setTimeFalling(float timeFalling) { timeFalling_ = timeFalling; }
  /// get the id
  constexpr inline HcalDetId id() const { return HcalDetId(detid()); }

  constexpr inline void setChiSquared(const float chi2) { chiSquared_ = chi2; }
  constexpr inline float chi2() const { return chiSquared_; }

  constexpr inline void setRawEnergy(const float en) { rawEnergy_ = en; }
  constexpr inline float eraw() const { return rawEnergy_; }

  constexpr inline void setAuxEnergy(const float en) { auxEnergy_ = en; }
  constexpr inline float eaux() const { return auxEnergy_; }

  constexpr inline void setAuxHBHE(const uint32_t aux) { auxHBHE_ = aux; }
  constexpr inline uint32_t auxHBHE() const { return auxHBHE_; }

  constexpr inline void setAuxPhase1(const uint32_t aux) { auxPhase1_ = aux; }
  constexpr inline uint32_t auxPhase1() const { return auxPhase1_; }

  constexpr inline void setAuxTDC(const uint32_t aux) { auxTDC_ = aux; }
  constexpr inline uint32_t auxTDC() const { return auxTDC_; }

  // The following method returns "true" for "Plan 1" merged rechits
  bool isMerged() const;

  // The following method fills the vector with the ids of the
  // rechits that have been merged to construct the "Plan 1" rechit.
  // For normal (i.e., not merged) rechits the vector will be cleared.
  void getMergedIds(std::vector<HcalDetId>* ids) const;

  // Returns the DetId of the front Id if it is a merged RecHit in "Plan 1"
  HcalDetId idFront() const;

private:
  float timeFalling_;
  float chiSquared_;
  float rawEnergy_;
  float auxEnergy_;
  uint32_t auxHBHE_;
  uint32_t auxPhase1_;
  uint32_t auxTDC_;
};

std::ostream& operator<<(std::ostream& s, const HBHERecHit& hit);

#endif
