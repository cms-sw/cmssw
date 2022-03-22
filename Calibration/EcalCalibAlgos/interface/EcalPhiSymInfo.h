#ifndef Calibration_EcalCalibAlgos_EcalPhiSymInfo_h
#define Calibration_EcalCalibAlgos_EcalPhiSymInfo_h

/** \class EcalPhiSymInfo
 * 
 * EcalPhiSym calibration lumi/run based information
 * 
 * Original Author: Simone Pigazzini (2022)
 */

#include <vector>
#include <cstdint>
#include <cassert>

class EcalPhiSymInfo {
public:
  //---ctors---
  EcalPhiSymInfo()
      : totHitsEB_(0),
        totHitsEE_(0),
        nEvents_(0),
        nLumis_(0),
        fillNumber_(0),
        delivLumi_(0),
        recLumi_(0),
        nMis_(0),
        minMisEB_(0),
        maxMisEB_(0),
        minMisEE_(0),
        maxMisEE_(0) {}

  EcalPhiSymInfo(
      uint64_t hitsEB, uint64_t hitsEE, uint64_t nEvents, uint32_t nLumis, uint16_t fill, float delivLumi, float recLumi)
      : totHitsEB_(hitsEB),
        totHitsEE_(hitsEE),
        nEvents_(nEvents),
        nLumis_(nLumis),
        fillNumber_(fill),
        delivLumi_(delivLumi),
        recLumi_(recLumi),
        nMis_(0),
        minMisEB_(0),
        maxMisEB_(0),
        minMisEE_(0),
        maxMisEE_(0) {}

  //---dtor---
  ~EcalPhiSymInfo() = default;

  //---setters---
  inline void setMiscalibInfo(
      const int& nmis, const float& minEB, const float& maxEB, const float& minEE, const float& maxEE) {
    nMis_ = nmis;
    minMisEB_ = minEB;
    maxMisEB_ = maxEB;
    minMisEE_ = minEE;
    maxMisEE_ = maxEE;
  };

  //---getters---
  inline uint64_t totHits() const { return totHitsEB_ + totHitsEE_; };
  inline uint64_t totHitsEB() const { return totHitsEB_; };
  inline uint64_t totHitsEE() const { return totHitsEE_; };
  inline uint32_t nEvents() const { return nEvents_; };
  inline uint16_t nLumis() const { return nLumis_; };
  inline uint16_t fillNumber() const { return fillNumber_; };
  inline float delivLumi() const { return delivLumi_; };
  inline float recLumi() const { return recLumi_; };
  inline uint8_t nMis() const { return nMis_; };
  inline float minMisEB() const { return minMisEB_; };
  inline float maxMisEB() const { return maxMisEB_; };
  inline float minMisEE() const { return minMisEE_; };
  inline float maxMisEE() const { return maxMisEE_; };

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
  uint8_t nMis_;
  float minMisEB_;
  float maxMisEB_;
  float minMisEE_;
  float maxMisEE_;
};

#endif
