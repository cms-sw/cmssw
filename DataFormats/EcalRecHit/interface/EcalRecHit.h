#ifndef DATAFORMATS_ECALRECHIT_H
#define DATAFORMATS_ECALRECHIT_H 1

#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"

/** \class EcalRecHit
 *  
 * $id: $
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
          kFake,                     // the signal in the channel is a fake (e.g. a so-called spike)
          kFakeNeighbours,           // the signal in the channel is a fake and it is detected by looking at the neighbours
          kDead,                     // channel is dead and any recovery fails
          kKilled,                   // MC only flag: the channel is killed in the real detector
          kTPSaturated,              // only for flagBits_: the channel is in a region with saturated TP
          kL1SpikeFlag,              // only for flagBits_: the channel is in a region with TP with sFGVB = 0
                                     // pro tempore, this will become obsolete when the online protection against spikes will be activated
                                     //
          kUnknown                   // to easy the interface with functions returning flags
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
  EcalRecHit(const DetId& id, float energy, float time, uint32_t flags = 0, uint32_t flagBits = 0);
  /// get the id
  // For the moment not returning a specific id for subdetector
  DetId id() const { return DetId(detid());}
  bool isRecovered() const;
  bool isTimeValid() const;
  bool isTimeErrorValid() const;
  uint32_t recoFlag() const { return 0xF & flags(); }
  float chi2Prob() const;          // not used
  float outOfTimeChi2Prob() const; // not used
  float chi2() const;
  float outOfTimeChi2() const;
  uint32_t flagBits() const { return flagBits_; }
  // set the energy for out of time events
  // (only energy >= 0 will be stored)
  float outOfTimeEnergy() const;
  float timeError() const;
  void setRecoFlag( uint32_t flag );
  void setChi2Prob( float chi2Prob );          // not used
  void setOutOfTimeChi2Prob( float chi2Prob ); // not used
  void setChi2( float chi2 );
  void setOutOfTimeChi2( float chi2 );
  void setOutOfTimeEnergy( float energy );
  void setFlagBits( uint32_t flagBits ) { flagBits_ = flagBits; }
  void setTimeError( uint8_t timeErrBits );

private:
  uint32_t flagBits_;
};

std::ostream& operator<<(std::ostream& s, const EcalRecHit& hit);

#endif
