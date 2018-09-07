#ifndef DIGIHCAL_HFDATAFRAME_H
#define DIGIHCAL_HFDATAFRAME_H

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/HcalDigi/interface/HcalQIESample.h"
#include <ostream>

/** \class HFDataFrame
       
Precision readout digi for HF

*/
class HFDataFrame {
public:
  typedef HcalDetId key_type; ///< For the sorted collection

  constexpr HFDataFrame() 
    : id_(0), size_(0), hcalPresamples_(0)
  {}
  constexpr explicit HFDataFrame(const HcalDetId& id) : 
    id_(id), size_(0), hcalPresamples_(0)
  {// TODO : test id for HcalForward
  }

  constexpr HcalDetId const& id() const { return id_; }
  constexpr HcalElectronicsId const& elecId() const { return electronicsId_; }
  
  /// total number of samples in the digi
  constexpr int size() const { return size_&0xF; }
  /// number of samples before the sample from the triggered beam crossing (according to the hardware)
  constexpr int presamples() const { return hcalPresamples_&0xF; }
   /// was ZS MarkAndPass?
  constexpr bool zsMarkAndPass() const { return (hcalPresamples_&0x10); }
  /// was ZS unsuppressed?
  constexpr bool zsUnsuppressed() const { return (hcalPresamples_&0x20); }
  /// zs crossing mask (which sums considered)
  constexpr uint32_t zsCrossingMask() const { return (hcalPresamples_&0x3FF000)>>12; }
  
  /// access a sample
  constexpr HcalQIESample const& operator[](int i) const { return data_[i]; }
  /// access a sample
  constexpr HcalQIESample const& sample(int i) const { return data_[i]; }

  /// offset of bunch number for this channel relative to nominal set in the unpacker (range is +7->-7.  -1000 indicates the data is invalid/unavailable)
  constexpr int fiberIdleOffset() const {
    int val=(hcalPresamples_&0xF00)>>8;
    return (val==0)?(-1000):(((val&0x8)==0)?(-(val&0x7)):(val&0x7));
  }
  
  /// validate appropriate DV and ER bits as well as capid rotation for the specified samples (default is all)
  constexpr bool validate(int firstSample=0, int nSamples=100) const {
    int capid=-1;
    bool ok=true;
    for (int i=0; ok && i<nSamples && i+firstSample<size_; i++) {
      if (data_[i+firstSample].er() || !data_[i+firstSample].dv()) ok=false;
      if (i==0) capid=data_[i+firstSample].capid();
      if (capid!=data_[i+firstSample].capid()) ok=false;
      capid=(capid+1)%4;
    }
    return ok;
  }
  
  constexpr void setSize(int size) {
    if (size>MAXSAMPLES) size_=MAXSAMPLES;
    else if (size<=0) size_=0;
    else size_=size;
  }
  constexpr void setPresamples(int ps) {
    hcalPresamples_|=ps&0xF;
  }
  constexpr void setZSInfo(bool unsuppressed, bool markAndPass, 
                              uint32_t crossingMask=0) {
    hcalPresamples_&=0x7FC00F0F; // preserve actual presamples and fiber idle offset
    if (markAndPass) hcalPresamples_|=0x10;
    if (unsuppressed) hcalPresamples_|=0x20;
    hcalPresamples_|=(crossingMask&0x3FF)<<12; 
  }
  constexpr void setSample(int i, const HcalQIESample& sam) { data_[i]=sam; }
  constexpr void setReadoutIds(const HcalElectronicsId& eid) {
    electronicsId_=eid;
  }
  constexpr void setFiberIdleOffset(int offset) {
    hcalPresamples_&=0x7FFFF0FF;
    if (offset>=7) hcalPresamples_|=0xF00;
    else if (offset>=0) hcalPresamples_|=(0x800)|(offset<<8);
    else if (offset>=-7) hcalPresamples_|=((-offset)<<8);
    else hcalPresamples_|=0x700;
  }
  
  static const int MAXSAMPLES = 10;
private:
  HcalDetId id_;
  HcalElectronicsId electronicsId_; 
  int size_;
  int hcalPresamples_;
  HcalQIESample data_[MAXSAMPLES];
};

std::ostream& operator<<(std::ostream&, const HFDataFrame&);

#endif
