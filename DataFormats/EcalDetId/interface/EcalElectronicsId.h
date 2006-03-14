#ifndef DATAFORMATS_ECALDETID_ECALELECTRONICSID_H
#define DATAFORMATS_ECALDETID_ECALELECTRONICSID_H 1

#include <boost/cstdint.hpp>
#include <ostream>

/** \brief Ecal readout channel identification
    [31:19] Unused (so far)
    [18:12]  DCC id
    [11:5]   tower
    [4:0]   channel
    Index starts from 1
 */


class EcalElectronicsId {
  
 public:

  /** Default constructor -- invalid value */
  EcalElectronicsId();
  /** from raw */
  EcalElectronicsId(uint32_t);
  /** Constructor from dcc,tower,channel **/
  EcalElectronicsId(int dccid, int towerid, int channelid);
  
  uint32_t operator()() { return EcalElectronicsId_; }
  uint32_t rawId() const { return EcalElectronicsId_; }
  
  /// get the DCC (Ecal Local DCC value not global one) id
  int dccId() const { return (EcalElectronicsId_>>12)&0x7F; }
  /// get the tower id
  int towerId() const { return (EcalElectronicsId_>>5)&0x7F; }
  /// get the channel id
  int channelId() const { return (EcalElectronicsId_&0x1F); }
  /// get a fast, compact, unique index for linear lookups (maximum value = 524287)
  int linearIndex() const { return (EcalElectronicsId_)&0x7FFFF; }
  
  static const int MAX_DCCID = 127; //To be updated with correct and final number
  static const int MIN_DCCID = 1;
  static const int MAX_TOWERID = 70;
  static const int MIN_TOWERID = 1;
  static const int MAX_CHANNELID = 25;
  static const int MIN_CHANNELID = 1;
  
  /** Equality operator */
  int operator==(const EcalElectronicsId& id) const { return id.EcalElectronicsId_==EcalElectronicsId_; }
  /** Non-Equality operator */
  int operator!=(const EcalElectronicsId& id) const { return id.EcalElectronicsId_!=EcalElectronicsId_; }
  /// Compare the id to another id for use in a map
  int operator<(const EcalElectronicsId& id) const { return EcalElectronicsId_<id.EcalElectronicsId_; }
  
  private:
  
  uint32_t EcalElectronicsId_;
  };
  
  std::ostream& operator<<(std::ostream&,const EcalElectronicsId&);

  
#endif
