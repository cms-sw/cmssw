#ifndef DATAFORMATS_ECALDETID_ECALELECTRONICSID_H
#define DATAFORMATS_ECALDETID_ECALELECTRONICSID_H 1

#include <ostream>
#include <cstdint>

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

/** \brief Ecal readout channel identification
    [32:20] Unused (so far)
    [19:13]  DCC id
    [12:6]   tower
    [5:3]    strip
    [2:0]    xtal
    Index starts from 1
 */

class EcalElectronicsId {
public:
  /** Default constructor -- invalid value */
  EcalElectronicsId();
  /** from raw */
  EcalElectronicsId(uint32_t);
  /** Constructor from dcc,tower,channel **/
  EcalElectronicsId(int dccid, int towerid, int stripid, int xtalid);

  uint32_t operator()() { return EcalElectronicsId_; }
  uint32_t rawId() const { return EcalElectronicsId_; }

  /// get the DCC (Ecal Local DCC value not global one) id
  int dccId() const { return (EcalElectronicsId_ >> 13) & 0x7F; }
  /// get the tower id
  int towerId() const { return (EcalElectronicsId_ >> 6) & 0x7F; }
  /// get the tower id
  int stripId() const { return (EcalElectronicsId_ >> 3) & 0x7; }
  /// get the channel id
  int xtalId() const { return (EcalElectronicsId_ & 0x7); }

  /// zside = +1 or -1
  int zside() const;

  /// get the subdet
  EcalSubdetector subdet() const;

  /// get a fast, compact, unique index for linear lookups (maximum value = 4194303)
  int linearIndex() const { return (EcalElectronicsId_)&0x3FFFFF; }

  /// so far for EndCap only :
  int channelId() const;  // xtal id between 1 and 25

  static const int MAX_DCCID = 54;  //To be updated with correct and final number
  static const int MIN_DCCID = 1;
  static const int MAX_TOWERID = 70;
  static const int MIN_TOWERID = 1;
  static const int MAX_STRIPID = 5;
  static const int MIN_STRIPID = 1;
  static const int MAX_CHANNELID = 25;
  static const int MIN_CHANNELID = 1;
  static const int MAX_XTALID = 5;
  static const int MIN_XTALID = 1;

  static const int MIN_DCCID_EEM = 1;
  static const int MAX_DCCID_EEM = 9;
  static const int MIN_DCCID_EBM = 10;
  static const int MAX_DCCID_EBM = 27;
  static const int MIN_DCCID_EBP = 28;
  static const int MAX_DCCID_EBP = 45;
  static const int MIN_DCCID_EEP = 46;
  static const int MAX_DCCID_EEP = 54;

  static const int DCCID_PHI0_EBM = 10;
  static const int DCCID_PHI0_EBP = 28;

  static const int kDCCChannelBoundary = 17;
  static const int DCC_EBM = 10;  // id of the DCC in EB- which contains phi=0 deg.
  static const int DCC_EBP = 28;  // id of the DCC in EB+ which contains phi=0 deg.
  static const int DCC_EEM = 1;   // id of the DCC in EE- which contains phi=0 deg.
  static const int DCC_EEP = 46;  // id of the DCC in EE+ which contains phi=0 deg.

  /** Equality operator */
  int operator==(const EcalElectronicsId& id) const { return id.EcalElectronicsId_ == EcalElectronicsId_; }
  /** Non-Equality operator */
  int operator!=(const EcalElectronicsId& id) const { return id.EcalElectronicsId_ != EcalElectronicsId_; }
  /// Compare the id to another id for use in a map
  int operator<(const EcalElectronicsId& id) const { return EcalElectronicsId_ < id.EcalElectronicsId_; }

private:
  uint32_t EcalElectronicsId_;
};

std::ostream& operator<<(std::ostream&, const EcalElectronicsId&);

#endif
