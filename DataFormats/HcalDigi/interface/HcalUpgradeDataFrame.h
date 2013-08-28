#ifndef DIGIHCAL_HCALUpgradeDATAFRAME_H
#define DIGIHCAL_HCALUpgradeDATAFRAME_H

#include <vector>
#include <ostream>
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/HcalDigi/interface/HcalUpgradeQIESample.h"

/** \class HcalUpgradeDataFrame
      
Precision readout digi for Upgrade work.

*/
class HcalUpgradeDataFrame {
public:
  typedef HcalDetId key_type; ///< For the sorted collection

  HcalUpgradeDataFrame(); 
  HcalUpgradeDataFrame(HcalDetId id);
  HcalUpgradeDataFrame(HcalDetId id, int capId, int samples, int presamples) ;

  const HcalDetId& id() const { return id_; }
  const HcalElectronicsId& elecId() const { return electronicsId_; }

  int size() const { return size_ ; }
  int presamples() const { return presamples_&0xF ; }
  bool zsMarkAndPass() const { return (presamples_&0x10); }
  bool zsUnsuppressed() const { return (presamples_&0x20); }
  uint32_t zsCrossingMask() const { return (presamples_&0x3FF000)>>12; }

  int startingCapId() const { return capId_ ; }
  int capId(int iSample=0) const { return (capId_+iSample)%4; }
  
  bool valid(int iSample=0) const { return dv_[iSample] ; }
  uint16_t adc(int iSample=0) const { return adc_[iSample] ; } 
  uint16_t tdc(int iSample=0) const { return tdc_[iSample] ; } 
  HcalUpgradeQIESample operator[](int iSample) const;

  void setSize(int size) ; 
  void setPresamples(int presamples) ;
  void setZSInfo(bool unsuppressed, bool markAndPass, uint32_t crossingMask=0);
  void setStartingCapId(int capId) { capId_ = capId ; } 
  void setSample(int iSample, uint16_t adc, uint16_t tdc, bool dv) ; 
  void setReadoutIds(const HcalElectronicsId& eid);

  static const int MAXSAMPLES = 10 ;
private:
  HcalDetId id_;
  HcalElectronicsId electronicsId_; 
  int capId_ ; 
  int size_, presamples_ ; 
  bool dv_[MAXSAMPLES] ;
  uint16_t adc_[MAXSAMPLES];
  uint16_t tdc_[MAXSAMPLES] ;
};

std::ostream& operator<<(std::ostream&, const HcalUpgradeDataFrame&) ;

#include "DataFormats/Common/interface/SortedCollection.h"
typedef edm::SortedCollection<HcalUpgradeDataFrame> HcalUpgradeDigiCollection;


#endif
