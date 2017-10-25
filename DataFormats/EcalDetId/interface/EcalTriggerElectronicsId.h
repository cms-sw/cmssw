#ifndef DATAFORMATS_ECALDETID_ECALTRIGGERELECTRONICSID_H
#define DATAFORMATS_ECALDETID_ECALTRIGGERELECTRONICSID_H 1

#include <ostream>
#include <cstdint>

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"


/** \brief Ecal trigger electronics identification
    [32:20] Unused (so far)
    [19:13]  TCC id
    [12:6]   TT id
    [5:3]    pseudo strip (in EB == strip)
    [2:0]    channel in pseudostrip 
    Index starts from 1
 */


class EcalTriggerElectronicsId {
  
 public:

  /** Default constructor -- invalid value */
  EcalTriggerElectronicsId();
  /** from raw */
  EcalTriggerElectronicsId(uint32_t);
  /** Constructor from tcc,tt,pseudostrip,channel **/
  EcalTriggerElectronicsId(int tccid, int towerid, int pseudostripid, int channelid);
  
  uint32_t operator()() { return EcalTriggerElectronicsId_; }
  uint32_t rawId() const { return EcalTriggerElectronicsId_; }
  
  /// get the DCC (Ecal Local DCC value not global one) id
  int tccId() const { return (EcalTriggerElectronicsId_>>13)&0x7F; }
  /// get the tower id
  int ttId() const { return (EcalTriggerElectronicsId_>>6)&0x7F; }
  /// get the tower id
  int pseudoStripId() const { return (EcalTriggerElectronicsId_>>3)&0x7; }
  /// get the channel id
  int channelId() const { return (EcalTriggerElectronicsId_&0x7); }
  /// get a fast, compact, unique index for linear lookups (maximum value = 1048575)
  int linearIndex() const { return (EcalTriggerElectronicsId_)&0xFFFFF; }
  
  /// get the zside() +1 / -1
  int zside() const;

  /// get the subdet
  EcalSubdetector subdet() const;

  static const int MAX_TCCID = 108; //To be updated with correct and final number
  static const int MIN_TCCID = 1;
  static const int MAX_TTID = 68;
  static const int MIN_TTID = 1;
  static const int MAX_PSEUDOSTRIPID = 5;
  static const int MIN_PSEUDOSTRIPID = 1;
  static const int MAX_CHANNELID = 5;
  static const int MIN_CHANNELID = 1;

  static const int MIN_TCCID_EEM = 1;
  static const int MAX_TCCID_EEM = 36;
  static const int MIN_TCCID_EBM = 37;
  static const int MAX_TCCID_EBM = 54;
  static const int MIN_TCCID_EBP = 55;
  static const int MAX_TCCID_EBP = 72;
  static const int MIN_TCCID_EEP = 73;
  static const int MAX_TCCID_EEP = 108;
  
  static const int TCCID_PHI0_EEM_IN  = 1;     // id of the inner TCC in EE- which contains phi=0 deg.
  static const int TCCID_PHI0_EEM_OUT = 19;    // id of the outer TCC in EE- which contains phi=0 deg.
  static const int TCCID_PHI0_EEP_IN  = 91;    // id of the inner TCC in EE+ which contains phi=0 deg.
  static const int TCCID_PHI0_EEP_OUT = 73;    // id of the outer TCC in EE+ which contains phi=0 deg.
  static const int TCCID_PHI0_EBM = 37;        // id of the TCC in EB- which contains phi=0 deg.
  static const int TCCID_PHI0_EBP = 55;        // id of the TCC in EB+ which contains phi=0 deg.


  /** Equality operator */
  int operator==(const EcalTriggerElectronicsId& id) const { return id.EcalTriggerElectronicsId_==EcalTriggerElectronicsId_; }
  /** Non-Equality operator */
  int operator!=(const EcalTriggerElectronicsId& id) const { return id.EcalTriggerElectronicsId_!=EcalTriggerElectronicsId_; }
  /// Compare the id to another id for use in a map
  int operator<(const EcalTriggerElectronicsId& id) const { return EcalTriggerElectronicsId_<id.EcalTriggerElectronicsId_; }
  
 private:
  
  uint32_t EcalTriggerElectronicsId_;
};

std::ostream& operator<<(std::ostream&,const EcalTriggerElectronicsId&);

  
#endif
