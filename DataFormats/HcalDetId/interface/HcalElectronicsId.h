#ifndef DATAFORMATS_HCALDETID_HCALELECTRONICSID_H
#define DATAFORMATS_HCALDETID_HCALELECTRONICSID_H 1

#include <string>
#include <ostream>
#include <cstdint>

/** \brief Readout chain identification for Hcal

    [31:27] Unused (so far)
    [26] VME (0), uTCA (1)

    For VME Electronics:
    [25]    Trigger-chain id flag
    [24:20] Readout Crate Id
    [19] HTR FPGA selector [t/b]
    [18:14] HTR Slot
    [13:9]  DCC id
    [8:5]   Spigot
    [4:2]   FiberIndex or SLB site
    [1:0]   FiberChanId or SLB channel

    For uTCA Electronics:

     [25]    Is Trigger Id
     [18:13] Readout Crate Id
     [12:9] Slot
     [8:4]   Fiber
     [3:0]   FiberChanId

 */
class HcalElectronicsId {
public:
  /** Default constructor -- invalid value */
  constexpr HcalElectronicsId() 
    : hcalElectronicsId_{0xffffffffu}
  {}
  /** from raw */
  constexpr HcalElectronicsId(uint32_t id)
    : hcalElectronicsId_{id}
  {}
  /** VME Constructor from fiberchan,fiber index,spigot,dccid */
  constexpr HcalElectronicsId(int fiberChan, int fiberIndex, int spigot, int dccid)
    : hcalElectronicsId_((uint32_t)((fiberChan&0x3) | (((fiberIndex-1)&0x7)<<2) |
      ((spigot&0xF)<<5) | ((dccid&0x1F)<<9)))
  {}
  /** VME Constructor from slb channel,slb site,spigot,dccid */
  constexpr HcalElectronicsId(int slbChan, int slbSite, int spigot, 
                              int dccid, int crate, int slot, int tb) 
    : hcalElectronicsId_((uint32_t)((slbChan&0x3) | (((slbSite)&0x7)<<2) |
      ((spigot&0xF)<<5) | ((dccid&0x1F)<<9))) {
    hcalElectronicsId_|=((tb&0x1)<<19) | ((slot&0x1f)<<14) | ((crate&0x3f)<<20);
    hcalElectronicsId_|=0x02000000;
  }
  /** uTCA constructor */
  constexpr HcalElectronicsId(int crate, int slot, int fiber, int fc, bool isTrigger) 
    : hcalElectronicsId_((int)((fc&0xF) | (((fiber)&0x1F)<<4) |
      ((slot&0xF)<<9) | ((crate&0x3F)<<13))) {
    if (isTrigger)   hcalElectronicsId_|=0x02000000;
    hcalElectronicsId_|=0x04000000;
  }

  constexpr uint32_t operator()() { return hcalElectronicsId_; }

  constexpr uint32_t rawId() const { return hcalElectronicsId_; }

  constexpr bool isVMEid() const { return (hcalElectronicsId_&0x04000000)==0; }
  constexpr bool isUTCAid() const { return (hcalElectronicsId_&0x04000000)!=0; }
  constexpr bool isTriggerChainId() const { return (hcalElectronicsId_&0x02000000)!=0; }

  /** Set the VME htr-related information 1=top, 0=bottom*/
  constexpr void setHTR(int crate, int slot, int tb) {
    if (isUTCAid()) return; // cannot do this for uTCA
    hcalElectronicsId_&=0x3FFF; // keep the readout chain info
    hcalElectronicsId_|=((tb&0x1)<<19) | ((slot&0x1f)<<14) | ((crate&0x3f)<<20);
  }

  /// get subtype for this channel (valid for uTCA only)
  constexpr int subtype() const { return (isUTCAid())?((hcalElectronicsId_>>21)&0x1F):(-1); }
  /// get the fiber channel id (which of channels on a fiber) 
  constexpr int fiberChanId() const { return (isVMEid())?(hcalElectronicsId_&0x3):(hcalElectronicsId_&0xF); }
  /// get the fiber index.  For VME 1-8 (which of eight fibers carried by a spigot), for uTCA fibers are zero-based
  constexpr int fiberIndex() const { return (isVMEid())?(((hcalElectronicsId_>>2)&0x7)+1):((hcalElectronicsId_>>4)&0x1F); }
  /// get the SLB channel index  (valid only for VME trigger-chain ids)
  constexpr int slbChannelIndex() const { return hcalElectronicsId_&0x3; }
  /// get the SLB site number (valid only for VME trigger-chain ids)
  constexpr int slbSiteNumber() const { return ((hcalElectronicsId_>>2)&0x7); }
 
  /// get the HTR channel id (1-24)
  constexpr int htrChanId() const { return isVMEid()?((fiberChanId()+1)+((fiberIndex()-1)*3)):(0); }

  /// get the HTR-wide slb channel code (letter plus number)
  std::string slbChannelCode() const;

  /// get the spigot (input number on DCC, AMC card number for uTCA)
  constexpr int spigot() const { return (isVMEid())?((hcalElectronicsId_>>5)&0xF):slot(); }
  /// get the (Hcal local) DCC id for VME, crate number for uTCA
  constexpr int dccid() const { return (isVMEid())?((hcalElectronicsId_>>9)&0x1F):crateId(); }
  /// get the htr slot
  constexpr int htrSlot() const { return slot(); }
  /// get the htr or uHTR slot
  constexpr int slot() const { return (isVMEid())?((hcalElectronicsId_>>14)&0x1F):((hcalElectronicsId_>>9)&0xF); }
  /// get the htr top/bottom (1=top/0=bottom), valid for VME
  constexpr int htrTopBottom() const { return (isVMEid())?((hcalElectronicsId_>>19)&0x1):(-1); }
  /// get the readout VME crate number
  constexpr int readoutVMECrateId() const { return crateId(); }
  /// get the readout VME crate number
  constexpr int crateId() const { return (isVMEid())?((hcalElectronicsId_>>20)&0x1F):((hcalElectronicsId_>>13)&0x3F); }
  /// get a fast, compact, unique index for linear lookups 
  constexpr int linearIndex() const { return (isVMEid())?((hcalElectronicsId_)&0x3FFF):((hcalElectronicsId_)&0x7FFFF); }

  static const int maxLinearIndex = 0x7FFFF; // 
  static const int maxDCCId = 31;
  
  /** Equality operator */
  constexpr int operator==(const HcalElectronicsId& id) const { return id.hcalElectronicsId_==hcalElectronicsId_; }
  /** Non-Equality operator */
  constexpr int operator!=(const HcalElectronicsId& id) const { return id.hcalElectronicsId_!=hcalElectronicsId_; }
  /// Compare the id to another id for use in a map
  constexpr int operator<(const HcalElectronicsId& id) const { return hcalElectronicsId_<id.hcalElectronicsId_; }

private:
  uint32_t hcalElectronicsId_;
};

std::ostream& operator<<(std::ostream&,const HcalElectronicsId&);


#endif
