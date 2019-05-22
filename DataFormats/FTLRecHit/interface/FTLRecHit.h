#ifndef DATAFORMATS_FTLRECHIT_H
#define DATAFORMATS_FTLRECHIT_H 1

#include "DataFormats/ForwardDetId/interface/MTDDetId.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"
#include <vector>

/** \class FTLRecHit
 *  
 * based on EcalRecHit
 *
 * \author Lindsey Gray
 */

class FTLRecHit {
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
  FTLRecHit(const DetId& id, float energy, float time, float timeError, uint32_t flagBits = 0);

  FTLRecHit(const DetId& id, uint8_t row, uint8_t column, float energy, 
	    float time, float timeError, uint32_t flagBits = 0);

  /// get the id

  float energy() const { return energy_; }
  void setEnergy(float energy) { energy_=energy; }
  
  const DetId& id() const { return id_; }
  const DetId& detid() const { return id(); }

  const MTDDetId mtdId() const { return MTDDetId(id_); }

  int row() const { return row_; }
  int column() const { return column_; }

  float time() const { return time_; }
  void setTime(float time) { time_=time; }

  bool isTimeValid() const;
  bool isTimeErrorValid() const;

  float timeError() const { return timeError_; }
  void setTimeError( float err ) { timeError_ = err; }
  
  /// set the flags (from Flags or ESFlags) 
  void setFlag(int flag) {flagBits_|= (0x1 << flag);}
  void unsetFlag(int flag) {flagBits_ &= ~(0x1 << flag);}

  /// check if the flag is true
  bool checkFlag(int flag) const{return flagBits_ & ( 0x1<<flag);}

  /// check if one of the flags in a set is true
  bool checkFlags(const std::vector<int>& flagsvec) const;


private:

  DetId id_;
  float energy_, time_, timeError_;
  uint8_t row_, column_;

  /// store rechit condition (see Flags enum) in a bit-wise way 
  unsigned char flagBits_;
};

std::ostream& operator<<(std::ostream& s, const FTLRecHit& hit);

#endif
