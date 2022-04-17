#ifndef Calibration_EcalCalibAlgos_EcalPhiSymRecHit_h
#define Calibration_EcalCalibAlgos_EcalPhiSymRecHit_h

/** \class EcalPhiSymRecHit
 * 
 * Dataformat dedicated to Phi Symmetry ecal calibration
 * 
 * Note: SumEt array ordering:
 *       0         - central value
 *       1<->N/2   - misCalib<1
 *       N/2+1<->N - misCalib>1
 *
 * Original Author: Simone Pigazzini (2022)
 */

#include <vector>
#include <cassert>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

class EcalPhiSymRecHit {
public:
  //---ctors---
  EcalPhiSymRecHit();
  EcalPhiSymRecHit(uint32_t id, unsigned int nMisCalibV, unsigned int status = 0);
  EcalPhiSymRecHit(uint32_t id, std::vector<float>& etValues, unsigned int status = 0);

  //---dtor---
  ~EcalPhiSymRecHit() = default;

  //---getters---
  inline uint32_t rawId() const { return id_; };
  inline int8_t eeRing() const { return eeRing_; };
  inline unsigned int statusCode() const { return chStatus_; };
  inline uint32_t nHits() const { return nHits_; };
  inline unsigned int nSumEt() const { return etSum_.size(); };
  inline float sumEt(int i = 0) const { return etSum_[i]; };
  inline float sumEt2() const { return et2Sum_; };
  inline float lcSum() const { return lcSum_; };
  inline float lc2Sum() const { return lc2Sum_; };

  //---setters---
  void setEERing(const int8_t& eering) { eeRing_ = eering; };

  //---utils---
  void addHit(const std::vector<float>& etValues, const float laserCorr = 0);
  void reset();

  //---operators---
  EcalPhiSymRecHit& operator+=(const EcalPhiSymRecHit& rhs);

private:
  uint32_t id_;
  int8_t eeRing_;
  unsigned int chStatus_;
  uint32_t nHits_;
  std::vector<float> etSum_;
  float et2Sum_;
  float lcSum_;
  float lc2Sum_;
};

typedef std::vector<EcalPhiSymRecHit> EcalPhiSymRecHitCollection;

#endif
