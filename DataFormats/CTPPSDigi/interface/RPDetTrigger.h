#ifndef DataFormats_CTPPSDigi_DET_TRIGGER_H
#define DataFormats_CTPPSDigi_DET_TRIGGER_H

#include "SimPPS/RPDigiProducer/interface/RPSimTypes.h"

class RPDetTrigger {
 public:
  RPDetTrigger(RPDetId det_id=0, unsigned short sector_no=0)
    {det_id_=det_id; sector_no_=sector_no;};
  inline RPDetId GetDetId() const {return det_id_;}
  inline unsigned short GetSector() const {return sector_no_;}
  
 private:
  RPDetId det_id_;
  unsigned short sector_no_;
};

// Comparison operators
inline bool operator<( const RPDetTrigger& one, const RPDetTrigger& other) {
  if(one.GetDetId() < other.GetDetId())
    return true;
  else if(one.GetDetId() == other.GetDetId())
    return one.GetSector() < other.GetSector();
  else 
    return false;
}

#endif
