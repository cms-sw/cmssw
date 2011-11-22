#ifndef DIGIHCAL_HCALUPGRADETRIGGERPRIMITIVEDIGI_H
#define DIGIHCAL_HCALUPGRADETRIGGERPRIMITIVEDIGI_H

#include <ostream>
#include <vector>
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDigi/interface/HcalUpgradeTriggerPrimitiveSample.h"

class HcalUpgradeTriggerPrimitiveDigi {

 public:
  
  //------------------------------------------------------
  // Needed for consistency with SortedCollection
  //------------------------------------------------------
  
  typedef HcalTrigTowerDetId key_type;
  explicit HcalUpgradeTriggerPrimitiveDigi(const HcalTrigTowerDetId& id);
  const HcalTrigTowerDetId& id() const { return m_id; }

  //------------------------------------------------------
  // Constructor/Destructor
  //------------------------------------------------------
  
  HcalUpgradeTriggerPrimitiveDigi(); 
  ~HcalUpgradeTriggerPrimitiveDigi();

  //------------------------------------------------------
  // Set information.  MAXSAMPLES sets size limit.
  //------------------------------------------------------

  void setSize      ( int  size );
  void setPresamples( int  presamples );
  void setZSInfo    ( bool unsuppressed, bool markAndPass);
  void setSample    ( int i, const HcalUpgradeTriggerPrimitiveSample& sample ) { m_data[i] = sample; }

  static const int MAXSAMPLES = 10;

  //------------------------------------------------------
  // Get the number of samples / presamples
  //------------------------------------------------------
  
  int size      () const { return (m_size           & 0xF); } // 
  int presamples() const { return (m_hcalPresamples & 0xF); } 
  
  //------------------------------------------------------
  // Get nformation about the ZS
  //------------------------------------------------------

  bool zsMarkAndPass () const { return m_hcalPresamples & 0x10; }
  bool zsUnsuppressed() const { return m_hcalPresamples & 0x20; }

  //------------------------------------------------------
  // Get information about individual samples
  //------------------------------------------------------
  
  // Access all stored samples

  const HcalUpgradeTriggerPrimitiveSample& operator[](int i) const { return m_data[i]; }
  const HcalUpgradeTriggerPrimitiveSample& sample    (int i) const { return m_data[i]; }
  
  // Access "sample of interest" directly
  
  const HcalUpgradeTriggerPrimitiveSample& t0() const { return m_data[presamples()]; }
  int SOI_fineGrain      () const { return t0().fineGrain      (); }
  int SOI_compressedEt   () const { return t0().compressedEt   (); }
  
 private:
  
  HcalTrigTowerDetId m_id;
  int m_size;
  int m_hcalPresamples;
  HcalUpgradeTriggerPrimitiveSample m_data [MAXSAMPLES];


};

#endif
