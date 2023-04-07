#ifndef DATAFORMATS_ECALRECHIT_H
#define DATAFORMATS_ECALRECHIT_H 1

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"

#include <vector>
#include <cmath>

/** \class EcalRecHit
 *  
 * \author P. Meridiani INFN Roma1
 */

class EcalRecHit {
public:
  typedef DetId key_type;

  // recHit flags
  enum Flags {
    kGood = 0,   // channel ok, the energy and time measurement are reliable
    kPoorReco,   // the energy is available from the UncalibRecHit, but approximate (bad shape, large chi2)
    kOutOfTime,  // the energy is available from the UncalibRecHit (sync reco), but the event is out of time
    kFaultyHardware,  // The energy is available from the UncalibRecHit, channel is faulty at some hardware level (e.g. noisy)
    kNoisy,           // the channel is very noisy
    kPoorCalib,  // the energy is available from the UncalibRecHit, but the calibration of the channel is poor
    kSaturated,  // saturated channel (recovery not tried)
    kLeadingEdgeRecovered,  // saturated channel: energy estimated from the leading edge before saturation
    kNeighboursRecovered,   // saturated/isolated dead: energy estimated from neighbours
    kTowerRecovered,        // channel in TT with no data link, info retrieved from Trigger Primitive
    kDead,                  // channel is dead and any recovery fails
    kKilled,                // MC only flag: the channel is killed in the real detector
    kTPSaturated,           // the channel is in a region with saturated TP
    kL1SpikeFlag,           // the channel is in a region with TP with sFGVB = 0
    kWeird,                 // the signal is believed to originate from an anomalous deposit (spike)
    kDiWeird,               // the signal is anomalous, and neighbors another anomalous signal
    kHasSwitchToGain6,      // at least one data frame is in G6
    kHasSwitchToGain1,      // at least one data frame is in G1
                            //
    kUnknown                // to ease the interface with functions returning flags.
  };

  // ES recHit flags
  enum ESFlags {
    kESGood,
    kESDead,
    kESHot,
    kESPassBX,
    kESTwoGoodRatios,
    kESBadRatioFor12,
    kESBadRatioFor23Upper,
    kESBadRatioFor23Lower,
    kESTS1Largest,
    kESTS3Largest,
    kESTS3Negative,
    kESSaturated,
    kESTS2Saturated,
    kESTS3Saturated,
    kESTS13Sigmas,
    kESTS15Sigmas
  };

  EcalRecHit() : energy_(0), time_(0), flagBits_(0) {}
  // by default a recHit is greated with no flag
  explicit EcalRecHit(const DetId& id, float energy, float time, uint32_t extra = 0, uint32_t flagBits = 0)
      : id_(id), energy_(energy), time_(time), flagBits_(flagBits), extra_(extra) {}

  float energy() const { return energy_; }
  void setEnergy(float energy) { energy_ = energy; }
  float time() const { return time_; }
  void setTime(float time) { time_ = time; }
  const DetId& detid() const { return id_; }

  /// get the id
  // For the moment not returning a specific id for subdetector
  // @TODO why this method?! should use detid()
  DetId id() const { return DetId(detid()); }

  bool isRecovered() const {
    return checkFlag(kLeadingEdgeRecovered) || checkFlag(kNeighboursRecovered) || checkFlag(kTowerRecovered);
  }

  bool isTimeValid() const { return (this->timeError() > 0); }

  bool isTimeErrorValid() const {
    if (!isTimeValid())
      return false;

    if (timeError() >= 10000)
      return false;

    return true;
  }

  static inline uint32_t getMasked(uint32_t value, uint32_t offset, uint32_t width) {
    return (value >> offset) & ((1 << width) - 1);
  }

  static inline uint32_t setMasked(uint32_t value, uint32_t x, uint32_t offset, uint32_t width) {
    const uint32_t mask = ((1 << width) - 1) << offset;
    value &= ~mask;
    value |= (x & ((1U << width) - 1)) << offset;
    return value;
  }

  static inline int getPower10(float e) {
    static constexpr float p10[] = {1.e-2f, 1.e-1f, 1.f, 1.e1f, 1.e2f, 1.e3f, 1.e4f, 1.e5f, 1.e6f};
    int b = e < p10[4] ? 0 : 5;
    for (; b < 9; ++b)
      if (e < p10[b])
        break;
    return b;
  }

  /* the new bit structure
   * 0..6   - chi2 in time events (chi2()), offset=0, width=7
   * 8..20  - energy uncertainty, offset=8, width=13
   * 24..31 - timeError(), offset=24, width=8
   */
  float chi2() const {
    uint32_t rawChi2 = getMasked(extra_, 0, 7);
    return (float)rawChi2 / (float)((1 << 7) - 1) * 64.;
  }

  void setChi2(float chi2) {
    // bound the max value of the chi2
    if (chi2 > 64)
      chi2 = 64;

    // use 7 bits
    uint32_t rawChi2 = lround(chi2 / 64. * ((1 << 7) - 1));
    extra_ = setMasked(extra_, rawChi2, 0, 7);
  }

  float energyError() const {
    uint32_t rawEnergy = getMasked(extra_, 8, 13);
    uint16_t exponent = rawEnergy >> 10;
    uint16_t significand = ~(0xE << 9) & rawEnergy;
    return (float)significand * pow(10, exponent - 5);
  }

  // set the energy uncertainty
  // (only energy >= 0 will be stored)
  void setEnergyError(float energy) {
    uint32_t rawEnergy = 0;
    if (energy > 0.001) {
      uint16_t exponent = getPower10(energy);
      static constexpr float ip10[] = {1.e5f, 1.e4f, 1.e3f, 1.e2f, 1.e1f, 1.e0f, 1.e-1f, 1.e-2f, 1.e-3f, 1.e-4};
      uint16_t significand = lround(energy * ip10[exponent]);
      // use 13 bits (3 exponent, 10 significand)
      rawEnergy = exponent << 10 | significand;
      /* here for doc and regression test
      uint16_t exponent_old = lround(floor(log10(energy))) + 3;  
      uint16_t significand_old = lround(energy/pow(10, exponent - 5));
      std::cout << energy << ' ' << exponent << ' ' << significand 
                          << ' ' << exponent_old <<	' ' << significand_old << std::endl;
      assert(exponent==exponent_old);
      assert(significand==significand_old);
      */
    }

    extra_ = setMasked(extra_, rawEnergy, 8, 13);
  }

  float timeError() const {
    uint32_t timeErrorBits = getMasked(extra_, 24, 8);
    // all bits off --> time reco bailed out (return negative value)
    if ((0xFF & timeErrorBits) == 0x00)
      return -1;
    // all bits on --> time error over 5 ns (return large value)
    if ((0xFF & timeErrorBits) == 0xFF)
      return 10000;

    float LSB = 1.26008;
    uint8_t exponent = timeErrorBits >> 5;
    uint8_t significand = timeErrorBits & ~(0x7 << 5);
    return pow(2., exponent) * significand * LSB / 1000.;
  }

  void setTimeError(uint8_t timeErrBits) { extra_ = setMasked(extra_, timeErrBits & 0xFF, 24, 8); }

  /// set the flags (from Flags or ESFlags)
  void setFlag(int flag) { flagBits_ |= (0x1 << flag); }
  void unsetFlag(int flag) { flagBits_ &= ~(0x1 << flag); }

  /// check if the flag is true
  bool checkFlag(int flag) const { return flagBits_ & (0x1 << flag); }

  /// check if one of the flags in a set is true
  bool checkFlags(const std::vector<int>& flagsvec) const {
    for (std::vector<int>::const_iterator flagPtr = flagsvec.begin(); flagPtr != flagsvec.end();
         ++flagPtr) {  // check if one of the flags is up

      if (checkFlag(*flagPtr))
        return true;
    }

    return false;
  }

  uint32_t flagsBits() const { return flagBits_; }

  /// apply a bitmask to our flags. Experts only
  bool checkFlagMask(uint32_t mask) const { return flagBits_ & mask; }

  /// DEPRECATED provided for temporary backward compatibility
  Flags recoFlag() const {
    for (int i = kUnknown;; --i) {
      if (checkFlag(i))
        return Flags(i);
      if (i == 0)
        break;
    }

    // no flag assigned, assume good
    return kGood;
  }

// For CC Timing reco
float nonCorrectedTime() const {

  // encoding constants in ns units
  float slope = 1.2;
  float offset = 16.0;
  float encoding = 31.875;

  uint8_t jitterErrorBits = getMasked(extra_, 24, 8); 
  float encodedBits = static_cast<float>( jitterErrorBits );
  float nonCorrectedTime = ( encodedBits > 1 && encodedBits < 254 ) ? (slope*time_ - encodedBits/encoding + offset) : -30.0;
  return nonCorrectedTime;
}

private:
  // from calorechit
  DetId id_;
  float energy_;
  float time_;

  /// store rechit condition (see Flags enum) in a bit-wise way
  uint32_t flagBits_;

  // packed uint32_t for timeError, chi2, energyError
  uint32_t extra_;
};

std::ostream& operator<<(std::ostream& s, const EcalRecHit& hit);

#endif
