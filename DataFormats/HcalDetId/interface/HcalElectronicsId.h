#ifndef DATAFORMATS_HCALDETID_HCALELECTRONICSID_H
#define DATAFORMATS_HCALDETID_HCALELECTRONICSID_H 1

#include <boost/cstdint.hpp>
#include <ostream>

/** \brief Readout chain identification for Hcal
    [31:25] Unused (so far)
    [24:20] Readout Crate Id
    [19] HTR FPGA selector [t/b]
    [18:14] HTR Slot
    [13:9]  DCC id
    [8:5]   Spigot
    [4:2]   FiberIndex
    [1:0]   FiberChanId
 */
class HcalElectronicsId {
public:
  /** Default constructor -- invalid value */
  HcalElectronicsId();
  /** from raw */
  HcalElectronicsId(uint32_t);
  /** Constructor from fiberchan,fiber index,spigot,dccid */
  HcalElectronicsId(int fiberChan, int fiberIndex, int spigot, int dccid);
  uint32_t operator()() { return hcalElectronicsId_; }


  /** Set the htr-related information 1=top, 0=bottom*/
  void setHTR(int crate, int slot, int tb);

  /// get the fiber channel id (which of three channels on a readout fiber)
  int fiberChanId() const { return hcalElectronicsId_&0x3; }
  /// get the fiber index (which of eight fibers carried by a spigot)
  int fiberIndex() const { return (hcalElectronicsId_>>2)&0x7; }
  /// get the HTR channel id (1-24)
  int htrChanId() const { return (fiberChanId()+1)+(fiberIndex()*3); }
  /// get the spigot (input number on DCC)
  int spigot() const { return (hcalElectronicsId_>>5)&0xF; }
  /// get the (Hcal local) DCC id
  int dccid() const { return (hcalElectronicsId_>>9)&0x1F; }
  /// get the htr slot
  int htrSlot() const { return (hcalElectronicsId_>>14)&0x1F; }
  /// get the htr top/bottom (1=top/0=bottom)
  int htrTopBottom() const { return (hcalElectronicsId_>>19)&0x1; }
  /// get the readout VME crate number
  int readoutVMECrateId() const { return (hcalElectronicsId_>>20)&0x1F; }
  /// get a fast, compact, unique index for linear lookups (maximum value = 16384)
  int linearIndex() const { return (hcalElectronicsId_)&0x3FFF; }

  static const int maxLinearIndex = 0x3FFF;
  static const int maxDCCId = 31;
  
  /** Equality operator */
  int operator==(const HcalElectronicsId& id) const { return id.hcalElectronicsId_==hcalElectronicsId_; }
  /** Non-Equality operator */
  int operator!=(const HcalElectronicsId& id) const { return id.hcalElectronicsId_!=hcalElectronicsId_; }
  /// Compare the id to another id for use in a map
  int operator<(const HcalElectronicsId& id) const { return hcalElectronicsId_<id.hcalElectronicsId_; }

private:
  uint32_t hcalElectronicsId_;
};

std::ostream& operator<<(std::ostream&,const HcalElectronicsId&);


#endif
