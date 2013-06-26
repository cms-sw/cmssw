#ifndef DIGIHCAL_HCALUPGRADEQIESAMPLE_H
#define DIGIHCAL_HCALUPGRADEQIESAMPLE_H

#include <ostream>
#include <boost/cstdint.hpp>

/** \class HcalUpgradeQIESample
 *  Simple container packer/unpacker for a single QIE data word
 *
 *
 *  $Date: 2013/03/27 14:55:41 $
 *  $Revision: 1.1 $
 *  \author J. Mans - Minnesota
 */
class HcalUpgradeQIESample {
public:
  HcalUpgradeQIESample() { theSample=0; }
  HcalUpgradeQIESample(uint16_t data) { theSample=data; }
  HcalUpgradeQIESample(int adc, int capid, int fiber, int fiberchan, bool dv=true, bool er=false);

  /// get the raw word
  uint16_t raw() const { return theSample; }
  /// get the ADC sample
  int adc() const { return theSample&0xFF; }
  /// get the nominal FC (no calibrations applied)
  double nominal_fC() const;
  /// get the Capacitor id
  int capid() const { return (theSample>>8)&0x3; }
  /// is the Data Valid bit set?
  bool dv() const { return (theSample&0x0400)!=0; }
  /// is the error bit set?
  bool er() const { return (theSample&0x0800)!=0; }
  /// get the fiber number
  int fiber() const { return ((theSample>>14)&0x7)+1; }
  /// get the fiber channel number
  int fiberChan() const { return (theSample>>12)&0x3; }
  /// get the id channel
  int fiberAndChan() const { return (theSample>>11)&0x1F; }  
  
  /// for streaming
  uint16_t operator()() { return theSample; }
  
private:
  uint16_t theSample;
};

std::ostream& operator<<(std::ostream&, const HcalUpgradeQIESample&);
  
#endif
