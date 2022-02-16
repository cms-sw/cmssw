#ifndef ECAL_PHISYM_INFO_H
#define ECAL_PHISYM_INFO_H

/** \class EcalPhiSymRecHit
 * 
 * EcalPhiSym calibration lumi/run based information
 * 
 * Original Author: Simone Pigazzini (2022)
 */

#include <vector>

class EcalPhiSymInfo {
public:
  //---ctors---
  EcalPhiSymInfo()
      : totHitsEB_(0), totHitsEE_(0), nEvents_(0), nLumis_(0), fillNumber_(0), delivLumi_(0), recLumi_(0) {}

  EcalPhiSymInfo(
      uint64_t hitsEB, uint64_t hitsEE, uint64_t nEvents, uint32_t nLumis, uint16_t fill, float delivLumi, float recLumi)
      : totHitsEB_(hitsEB),
        totHitsEE_(hitsEE),
        nEvents_(nEvents),
        nLumis_(nLumis),
        fillNumber_(fill),
        delivLumi_(delivLumi),
        recLumi_(recLumi) {}

  //---dtor---
  ~EcalPhiSymInfo(){};

  //---getters---
  inline uint64_t totHits() const { return totHitsEB_ + totHitsEE_; };
  inline uint64_t totHitsEB() const { return totHitsEB_; };
  inline uint64_t totHitsEE() const { return totHitsEE_; };
  inline uint32_t nEvents() const { return nEvents_; };
  inline uint16_t nLumis() const { return nLumis_; };
  inline uint16_t fillNumber() const { return fillNumber_; };
  inline float delivLumi() const { return delivLumi_; };
  inline float recLumi() const { return recLumi_; };

  //---operators---
  EcalPhiSymInfo& operator+=(const EcalPhiSymInfo& rhs) {
    // The class at the moment is designed to
    // hold at most data from a single run.
    // This implies fillNumber has to be the same,
    // unless it was not set, in that case it is 0.
    if (fillNumber_ != 0 && rhs.fillNumber() != 0)
      assert(fillNumber_ == rhs.fillNumber());
    else
      fillNumber_ = std::max(fillNumber_, rhs.fillNumber());
    totHitsEB_ += rhs.totHitsEB();
    totHitsEE_ += rhs.totHitsEE();
    nEvents_ += rhs.nEvents();
    nLumis_ += rhs.nLumis();
    delivLumi_ += rhs.delivLumi();
    recLumi_ += rhs.recLumi();

    return *this;
  }

private:
  uint64_t totHitsEB_;
  uint64_t totHitsEE_;
  uint32_t nEvents_;
  uint16_t nLumis_;
  uint16_t fillNumber_;
  float delivLumi_;
  float recLumi_;
};

#endif
