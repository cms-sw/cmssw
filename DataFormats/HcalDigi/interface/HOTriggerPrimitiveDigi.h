#ifndef DIGIHCAL_HOTRIGGERPRIMITIVEDIGI_H
#define DIGIHCAL_HOTRIGGERPRIMITIVEDIGI_H

#include <boost/cstdint.hpp>
#include <ostream>
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

/** \class HOTriggerprimitiveDigi
 *  Simple container packer/unpacker for a single 
 *  Trigger Primitive from an HO HTR
 *
 *
 *  $Date: 2008/04/23 00:53:12 $
 *  $Revision: 1.3 $
 *  \author J. St. John - Boston U
 */
class HOTriggerPrimitiveDigi {
public:
  typedef HcalDetId key_type; ///< For the sorted collection
 
  HOTriggerPrimitiveDigi() { theHO_TP=0; }
  HOTriggerPrimitiveDigi(uint32_t data) { theHO_TP=data; }
  HOTriggerPrimitiveDigi(int ieta, int iphi, int nsamples, int whichSampleTriggered, int databits);
 
  const HcalDetId id() const { return HcalDetId(HcalOuter,ieta(),iphi(),4); }
 
  /// get the raw (packed) Triger Primitive
  uint32_t raw() const { return theHO_TP; }
  /// get the raw ieta value 
  int raw_ieta() const { return theHO_TP&0x1F; }
  /// get the sign of ieta (int: +/- 1)
  int ieta_sign() const { return ((theHO_TP&0x10)?(-1):(1)); }
  /// get the absolute value of ieta
  int ieta_abs() const { return (theHO_TP&0x000F); }
  /// get the signed ieta value 
  int ieta() const { return ieta_abs()*ieta_sign(); }
  /// get the iphi value
  int iphi() const { return (theHO_TP>>5)&0x007F; }
  /// get the number of samples used to compute the TP
  int nsamples() const { return (theHO_TP>>12)&0x000F; }
  /// get the number of the triggering sample
  int whichSampleTriggered() const { return (theHO_TP>>16)&0x000F; }
  /// get the single-bit data
  int bits() const { return (theHO_TP>>20)&0x03FF; }

  static const int HO_TP_SAMPLES_MAX = 10;

  /// get one bit from the single-bit data.
  /// required to be called with a legal value.
  bool data(int whichbit=HO_TP_SAMPLES_MAX) const; 
      
  /// for streaming
  uint32_t operator()() { return theHO_TP; }
  
private:
  uint32_t theHO_TP;
};

std::ostream& operator<<(std::ostream&, const HOTriggerPrimitiveDigi&);
  
#endif
