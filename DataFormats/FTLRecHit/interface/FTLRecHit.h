#ifndef DATAFORMATS_FTLRECHIT_H
#define DATAFORMATS_FTLRECHIT_H 1

#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"
#include <vector>

/** \class FTLRecHit
 *  
 * based on EcalRecHit
 *
 * \author Lindsey Gray
 */

class FTLRecHit : public CaloRecHit {
public:
  typedef DetId key_type;

  // FTLEE recHit flags
    enum Flags { 
          kGood=0,                   // channel ok, the energy and time measurement are reliable          
          kKilled,                   // MC only flag: the channel is killed in the real detector
          kUnknown                   // to ease the interface with functions returning flags. 
  };
    
  /** bit structure of CaloRecHit::flags_ used in FTLRecHit:
   *
   *  | 32 | 31...25 | 24...12 | 11...5 | 4...1 |
   *     |      |         |         |       |
   *     |      |         |         |       +--> reco flags       ( 4 bits)
   *     |      |         |         +--> chi2 for in time events  ( 7 bits)
   *     |      |         +--> energy for out-of-time events      (13 bits)
   *     |      +--> chi2 for out-of-time events                  ( 7 bits)
   *     +--> spare                                               ( 1 bit )
   */

  FTLRecHit();
  // by default a recHit is greated with no flag
  FTLRecHit(const DetId& id, float energy, float time, uint32_t flags = 0, uint32_t flagBits = 0);
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

std::ostream& operator<<(std::ostream& s, const FTLRecHit& hit);

#endif
