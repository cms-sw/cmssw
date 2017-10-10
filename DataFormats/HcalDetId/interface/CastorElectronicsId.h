#ifndef DATAFORMATS_HCALDETID_CASTORELECTRONICSID_H
#define DATAFORMATS_HCALDETID_CASTORELECTRONICSID_H 1

#include <string>
#include <ostream>
#include <cstdint>

/** \brief Readout chain identification for Castor 
Bits for the readout chain : some names need change!
    [31:26] not used
    [25]    
    [24:20] 
    [19] 
    [18:14] 
    [13:9]  
    [8:5]   
    [4:2]   
    [1:0]   
 */
class CastorElectronicsId {
public:
  /** Constructors */
  CastorElectronicsId();
  CastorElectronicsId(uint32_t);
  CastorElectronicsId(int fiberChan, int fiberIndex, int spigot, int dccid);
  CastorElectronicsId(int slbChan, int slbSite, int spigot, int dccid, int crate, int slot, int tb);
  uint32_t operator()() { return castorElectronicsId_; }

  uint32_t rawId() const { return castorElectronicsId_; }

  bool isTriggerChainId() const { return (castorElectronicsId_&0x02000000)!=0; }

  void setHTR(int crate, int slot, int tb);
  int fiberChanId() const { return castorElectronicsId_&0x3; }
  int fiberIndex() const { return ((castorElectronicsId_>>2)&0xf)+1; }
  int slbChannelIndex() const { return castorElectronicsId_&0x3; }
  int slbSiteNumber() const { return ((castorElectronicsId_>>2)&0xf)+1; }

  std::string slbChannelCode() const;

  int htrChanId() const { return (fiberChanId()+1)+((fiberIndex()-1)*3); }
  int spigot() const { return (castorElectronicsId_>>6)&0xF; }
  int dccid() const { return (castorElectronicsId_>>10)&0xF; }
  int htrSlot() const { return (castorElectronicsId_>>14)&0x1F; }
  int htrTopBottom() const { return (castorElectronicsId_>>19)&0x1; }
  int readoutVMECrateId() const { return (castorElectronicsId_>>20)&0x1F; }
  int linearIndex() const { return (castorElectronicsId_)&0x3FFF; }

  static const int maxLinearIndex = 0x3FFF;
  static const int maxDCCId = 15;
  
  /** operators */
  int operator==(const CastorElectronicsId& id) const { return id.castorElectronicsId_==castorElectronicsId_; }
  int operator!=(const CastorElectronicsId& id) const { return id.castorElectronicsId_!=castorElectronicsId_; }
  /// Compare the id to another one for use in a map
  int operator<(const CastorElectronicsId& id) const { return castorElectronicsId_<id.castorElectronicsId_; }

private:
  uint32_t castorElectronicsId_;
};

std::ostream& operator<<(std::ostream&,const CastorElectronicsId&);


#endif
