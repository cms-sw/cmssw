/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*
****************************************************************************/

#ifndef DataFormats_TotemRPReco_RPDetTrigger
#define DataFormats_TotemRPReco_RPDetTrigger

class RPDetTrigger {
 public:
  RPDetTrigger(unsigned int det_id=0, unsigned short sector_no=0)
    {det_id_=det_id; sector_no_=sector_no;};
  inline unsigned int GetDetId() const {return det_id_;}
  inline unsigned short GetSector() const {return sector_no_;}
  
 private:
  unsigned int det_id_;
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
