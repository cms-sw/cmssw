#ifndef DIGIHCAL_HCALUpgradeDATAFRAME_H
#define DIGIHCAL_HCALUpgradeDATAFRAME_H

#include <vector>
#include <ostream>
#include "DataFormats/HcalDetId/interface/HcalDetId.h"


/** \class HcalUpgradeDataFrame
      
Precision readout digi for Upgrade work.

*/
class HcalUpgradeDataFrame {
public:
  typedef HcalDetId key_type; ///< For the sorted collection

  HcalUpgradeDataFrame(); 
  HcalUpgradeDataFrame(HcalDetId id, int capId, int samples, int presamples) ;

  const HcalDetId& id() const { return id_; }

  int size() const { return size_ ; }
  int presamples() const { return presamples_ ; }
  int startingCapId() const { return capId_ ; }
  int capId(int iSample=0) const { return (capId_+iSample)%4; }
  
  bool valid(int iSample=0) const { return dv_[iSample] ; }
  uint16_t adc(int iSample=0) const { return adc_[iSample] ; } 
  uint8_t tdc(int iSample=0) const { return tdc_[iSample] ; } 
  
  void setSize(int size) ; 
  void setPresamples(int presamples) ;
  void setStartingCapId(int capId) { capId_ = capId ; } 
  void setSample(int relSample, const uint16_t adc, const uint8_t tdc, const bool dv) ; 
  
  static const int MAXSAMPLES = 10 ;
private:
  HcalDetId id_;
  int capId_ ; 
  int size_, presamples_ ; 
  bool dv_[MAXSAMPLES] ;
  uint16_t adc_[MAXSAMPLES];
  uint8_t tdc_[MAXSAMPLES] ;
};

std::ostream& operator<<(std::ostream&, const HcalUpgradeDataFrame&) ;

#include "DataFormats/Common/interface/SortedCollection.h"
typedef edm::SortedCollection<HcalUpgradeDataFrame> HcalUpgradeDigiCollection;


#endif
