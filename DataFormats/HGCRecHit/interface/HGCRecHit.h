#ifndef DATAFORMATS_HGCRECHIT_H
#define DATAFORMATS_HGCRECHIT_H 1

#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"
#include <vector>

/** \class HGCRecHit
 *  
 * based on EcalRecHit
 *
 * \author Valeri Andreev
 */

class HGCRecHit : public CaloRecHit {
public:
  typedef DetId key_type;

  // HGCEE recHit flags
    enum Flags { 
          kGood=0,                   // channel ok, the energy and time measurement are reliable
          kPoorReco,                 // the energy is available from the UncalibRecHit, but approximate (bad shape, large chi2)
          kOutOfTime,                // the energy is available from the UncalibRecHit (sync reco), but the event is out of time
          kFaultyHardware,           // The energy is available from the UncalibRecHit, channel is faulty at some hardware level (e.g. noisy)
          kNoisy,                    // the channel is very noisy
          kPoorCalib,                // the energy is available from the UncalibRecHit, but the calibration of the channel is poor
          kSaturated,                // saturated channel (recovery not tried)
          kDead,                     // channel is dead and any recovery fails
          kKilled,                   // MC only flag: the channel is killed in the real detector
          kWeird,                    // the signal is believed to originate from an anomalous deposit (spike) 
          kDiWeird,                  // the signal is anomalous, and neighbors another anomalous signal  
                                     //
          kUnknown                   // to ease the interface with functions returning flags. 
  };

  //  HGCfhe recHit flags
  enum HGCfheFlags {
          kHGCfheGood,
          kHGCfheDead,
          kHGCfheHot,
          kHGCfhePassBX,
          kHGCfheSaturated
  };

  //  HGCbhe recHit flags
  enum HGCbheFlags {
          kHGCbheGood,
          kHGCbheDead,
          kHGCbheHot,
          kHGCbhePassBX,
          kHGCbheSaturated
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

  HGCRecHit();
  // by default a recHit is greated with no flag
  HGCRecHit(const DetId& id, float energy, float time, uint32_t flags = 0, uint32_t flagBits = 0);
  /// get the id
  // For the moment not returning a specific id for subdetector
  DetId id() const { return DetId(detid());}
  /////  bool isRecovered() const;
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


private:

  /// store rechit condition (see Flags enum) in a bit-wise way 
  uint32_t flagBits_;
};

std::ostream& operator<<(std::ostream& s, const HGCRecHit& hit);

#endif
