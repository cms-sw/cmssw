#ifndef DIGIHCAL_ZDCDATAFRAME_H
#define DIGIHCAL_ZDCDATAFRAME_H

#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/HcalDigi/interface/HcalQIESample.h"
#include <vector>
#include <ostream>

/** \class ZDCDataFrame
       
Precision readout digi for ZDC

*/
class ZDCDataFrame {
public:
  typedef HcalZDCDetId key_type; ///< For the sorted collection

  ZDCDataFrame(); // for persistence
  explicit ZDCDataFrame(const HcalZDCDetId& id);
  
  const HcalZDCDetId& id() const { return id_; }
  const HcalElectronicsId& elecId() const { return electronicsId_; }
  
  /// total number of samples in the digi
  int size() const { return size_&0xF; }
  /// number of samples before the sample from the triggered beam crossing (according to the hardware)
  int presamples() const { return hcalPresamples_&0xF; }
  /// was ZS MarkAndPass?
  bool zsMarkAndPass() const { return (hcalPresamples_&0x10); }
  /// was ZS unsuppressed?
  bool zsUnsuppressed() const { return (hcalPresamples_&0x20); }
  /// zs crossing mask (which sums considered)
  uint32_t zsCrossingMask() const { return (hcalPresamples_&0x3FF000)>>12; }
  
  /// access a sample
  const HcalQIESample& operator[](int i) const { return data_[i]; }
  /// access a sample
  const HcalQIESample& sample(int i) const { return data_[i]; }

  /// offset of bunch number for this channel relative to nominal set in the unpacker (range is +7->-7.  -1000 indicates the data is invalid/unavailable)
  int fiberIdleOffset() const;
  
  /// validate appropriate DV and ER bits as well as capid rotation for the specified samples (default is all)
  bool validate(int firstSample=0, int nSamples=100) const;
  
  void setSize(int size);
  void setPresamples(int ps);
  void setZSInfo(bool unsuppressed, bool markAndPass, uint32_t crossingMask=0);
  void setSample(int i, const HcalQIESample& sam) { data_[i]=sam; }
  void setReadoutIds(const HcalElectronicsId& eid);
  void setFiberIdleOffset(int offset);
  
  static const int MAXSAMPLES = 10;
private:
  HcalZDCDetId id_;
  HcalElectronicsId electronicsId_; 
  int size_;
  int hcalPresamples_; // also contains information about ZS MarkAndPass
  HcalQIESample data_[MAXSAMPLES];
};

std::ostream& operator<<(std::ostream&, const ZDCDataFrame&);

#endif
