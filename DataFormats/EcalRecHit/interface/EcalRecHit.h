#ifndef DATAFORMATS_ECALRECHIT_H
#define DATAFORMATS_ECALRECHIT_H 1

#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"
#include <vector>

/** \class EcalRecHit
 *  
 * $Id: EcalRecHit.h,v 1.24 2012/01/30 16:03:39 theofil Exp $
 * \author P. Meridiani INFN Roma1
 */

class EcalRecHit : public CaloRecHit {
public:
  typedef DetId key_type;

  // recHit flags
    enum Flags { 
          kGood=0,                   // channel ok, the energy and time measurement are reliable
          kPoorReco,                 // the energy is available from the UncalibRecHit, but approximate (bad shape, large chi2)
          kOutOfTime,                // the energy is available from the UncalibRecHit (sync reco), but the event is out of time
          kFaultyHardware,           // The energy is available from the UncalibRecHit, channel is faulty at some hardware level (e.g. noisy)
          kNoisy,                    // the channel is very noisy
          kPoorCalib,                // the energy is available from the UncalibRecHit, but the calibration of the channel is poor
          kSaturated,                // saturated channel (recovery not tried)
          kLeadingEdgeRecovered,     // saturated channel: energy estimated from the leading edge before saturation
          kNeighboursRecovered,      // saturated/isolated dead: energy estimated from neighbours
          kTowerRecovered,           // channel in TT with no data link, info retrieved from Trigger Primitive
          kDead,                     // channel is dead and any recovery fails
          kKilled,                   // MC only flag: the channel is killed in the real detector
          kTPSaturated,              // the channel is in a region with saturated TP
          kL1SpikeFlag,              // the channel is in a region with TP with sFGVB = 0
          kWeird,                    // the signal is believed to originate from an anomalous deposit (spike) 
          kDiWeird,                  // the signal is anomalous, and neighbors another anomalous signal  
          kHasSwitchToGain6,         // at least one data frame is in G6
          kHasSwitchToGain1,         // at least one data frame is in G1
                                     //
          kUnknown                   // to ease the interface with functions returning flags. 
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

  /** bit structure of CaloRecHit::flags_ used in EcalRecHit:
   *
   *  | 32 | 31...25 | 24...12 | 11...5 | 4...1 |
   *     |      |         |         |       |
   *     |      |         |         |       +--> reco flags       ( 4 bits)
   *     |      |         |         +--> chi2 for in time events  ( 7 bits)
   *     |      |         +--> energy for out-of-time events      (13 bits)
   *     |      +--> chi2 for out-of-time events                  ( 7 bits)
   *     +--> spare                                               ( 1 bit )
   */

  EcalRecHit();
  // by default a recHit is greated with no flag
  EcalRecHit(const DetId& id, float energy, float time, uint32_t flags = 0, uint32_t flagBits = 0);
  /// get the id
  // For the moment not returning a specific id for subdetector
  DetId id() const { return DetId(detid());}
  bool isRecovered() const;
  bool isTimeValid() const;
  bool isTimeErrorValid() const;


  float chi2() const;
  float outOfTimeChi2() const;

  // set the energy for out of time events
  // (only energy >= 0 will be stored)
  float outOfTimeEnergy() const;
  float timeError() const;

  void setChi2( float chi2 );
  void setOutOfTimeChi2( float chi2 );
  void setOutOfTimeEnergy( float energy );

  void setTimeError( uint8_t timeErrBits );

  
  /// set the flags (from Flags or ESFlags) 
  void setFlag(int flag) {flagBits_|= (0x1 << flag);}
  void unsetFlag(int flag) {flagBits_ &= ~(0x1 << flag);}

  /// check if the flag is true
  bool checkFlag(int flag) const{return flagBits_ & ( 0x1<<flag);}

  /// check if one of the flags in a set is true
  bool checkFlags(const std::vector<int>& flagsvec) const;

  /// apply a bitmask to our flags. Experts only
  bool checkFlagMask(uint32_t mask) const { return flagBits_&mask; }

  /// DEPRECATED provided for temporary backward compatibility
  Flags recoFlag() const ;

private:

  /// store rechit condition (see Flags enum) in a bit-wise way 
  uint32_t flagBits_;
};

std::ostream& operator<<(std::ostream& s, const EcalRecHit& hit);

#endif
