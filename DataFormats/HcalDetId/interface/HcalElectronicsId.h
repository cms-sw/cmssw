#ifndef DATAFORMATS_HCALDETID_HCALELECTRONICSID_H
#define DATAFORMATS_HCALDETID_HCALELECTRONICSID_H 1

#include <string>
#include <ostream>
#include <stdint.h>

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

     [25:21] Subtype
     [17:13] Readout Crate Id
     [12:9] Slot
     [8:4]   Fiber
     [3:0]   FiberChanId

     Subtypes:
        0x01 = 1.6 Gbps precision (3 channel)
        0x02 = 4.8 Gbps 6 channel ADC+RETDC
        0x03 = 4.8 Gbps 12 channel FETDC
        0x04 = 4.8 Gbps 4 channel ADC+RETDC+FETDC
        0x11 = optical RCT trigger
        0x12 = 6.4 Gbps HF trigger
        0x13 = 4.8 Gbps HB/HE trigger
        0x14 = 6.4 Gbps HB/HE trigger
 */
class HcalElectronicsId {
public:
  /** Default constructor -- invalid value */
  HcalElectronicsId();
  /** from raw */
  HcalElectronicsId(uint32_t);
  /** VME Constructor from fiberchan,fiber index,spigot,dccid */
  HcalElectronicsId(int fiberChan, int fiberIndex, int spigot, int dccid);
  /** VME Constructor from slb channel,slb site,spigot,dccid */
  HcalElectronicsId(int slbChan, int slbSite, int spigot, int dccid, int crate, int slot, int tb);
  /** uTCA constructor */
  HcalElectronicsId(int subtype, int crate, int slot, int fiber, int fiberchan);

  uint32_t operator()() { return hcalElectronicsId_; }

  uint32_t rawId() const { return hcalElectronicsId_; }

  bool isVMEid() const { return (hcalElectronicsId_&0x04000000)==0; }
  bool isUTCAid() const { return (hcalElectronicsId_&0x04000000)!=0; }
  bool isTriggerChainId() const { return (hcalElectronicsId_&0x02000000)!=0; }

  /** Set the VME htr-related information 1=top, 0=bottom*/
  void setHTR(int crate, int slot, int tb);

  /// get subtype for this channel (valid for uTCA only)
  int subtype() const { return (isUTCAid())?((hcalElectronicsId_>>21)&0x1F):(-1); }
  /// get the fiber channel id (which of channels on a fiber) 
  int fiberChanId() const { return (isVMEid())?(hcalElectronicsId_&0x3):(hcalElectronicsId_&0xF); }
  /// get the fiber index.  For VME 1-8 (which of eight fibers carried by a spigot), for uTCA fibers are zero-based
  int fiberIndex() const { return (isVMEid())?(((hcalElectronicsId_>>2)&0x7)+1):((hcalElectronicsId_>>4)&0x1F); }
  /// get the SLB channel index  (valid only for VME trigger-chain ids)
  int slbChannelIndex() const { return hcalElectronicsId_&0x3; }
  /// get the SLB site number (valid only for VME trigger-chain ids)
  int slbSiteNumber() const { return ((hcalElectronicsId_>>2)&0x7); }
 
  /// get the HTR channel id (1-24)
  int htrChanId() const { return isVMEid()?((fiberChanId()+1)+((fiberIndex()-1)*3)):(0); }

  /// get the HTR-wide slb channel code (letter plus number)
  std::string slbChannelCode() const;

  /// get the spigot (input number on DCC, AMC card number for uTCA)
  int spigot() const { return (isVMEid())?((hcalElectronicsId_>>5)&0xF):slot(); }
  /// get the (Hcal local) DCC id for VME, crate number for uTCA
  int dccid() const { return (isVMEid())?((hcalElectronicsId_>>9)&0x1F):crateId(); }
  /// get the htr slot
  int htrSlot() const { return slot(); }
  /// get the htr or uHTR slot
  int slot() const { return (isVMEid())?((hcalElectronicsId_>>14)&0x1F):((hcalElectronicsId_>>9)&0xF); }
  /// get the htr top/bottom (1=top/0=bottom), valid for VME
  int htrTopBottom() const { return (isVMEid())?((hcalElectronicsId_>>19)&0x1):(-1); }
  /// get the readout VME crate number
  int readoutVMECrateId() const { return crateId(); }
  /// get the readout VME crate number
  int crateId() const { return (isVMEid())?((hcalElectronicsId_>>20)&0x1F):((hcalElectronicsId_>>13)&0x1F); }
  /// get a fast, compact, unique index for linear lookups (maximum value = 16384)
  int linearIndex() const { return (isVMEid())?((hcalElectronicsId_)&0x3FFF):((hcalElectronicsId_)&0x3FFFF); }

  static const int maxLinearIndex = 0x3FFFF; // 
  static const int maxDCCId = 31;
  
  /** Equality operator */
  int operator==(const HcalElectronicsId& id) const { return id.hcalElectronicsId_==hcalElectronicsId_; }
  /** Non-Equality operator */
  int operator!=(const HcalElectronicsId& id) const { return id.hcalElectronicsId_!=hcalElectronicsId_; }
  /// Compare the id to another id for use in a map
  int operator<(const HcalElectronicsId& id) const { return hcalElectronicsId_<id.hcalElectronicsId_; }

  static const int st_PRECISION_1_6              = 0x01;
  static const int st_PRECISION_4_8_6CHAN        = 0x02;
  static const int st_PRECISION_4_8_12CHAN_TDC   = 0x03;
  static const int st_PRECISION_4_8_4CHAN        = 0x04;
  static const int st_TRIGGER_RCT                = 0x11;
  static const int st_TRIGGER_HF_6_4             = 0x12;
  static const int st_TRIGGER_HBHE_4_8           = 0x13;
  static const int st_TRIGGER_HBHE_6_4           = 0x14;

private:
  uint32_t hcalElectronicsId_;
};

std::ostream& operator<<(std::ostream&,const HcalElectronicsId&);


#endif
