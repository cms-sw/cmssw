#include "DataFormats/HcalDigi/interface/HcalUpgradeDataFrame.h"

HcalUpgradeDataFrame::HcalUpgradeDataFrame() : id_(0), 
					       electronicsId_(0),
					       capId_(0),
					       size_(0),
					       presamples_(0)
{
}

HcalUpgradeDataFrame::HcalUpgradeDataFrame(HcalDetId id) : id_(id),
							   electronicsId_(0),
							   capId_(0),
							   size_(0),
							   presamples_(0) 
{
}

HcalUpgradeDataFrame::HcalUpgradeDataFrame(HcalDetId id, int capId, int samples, int presamples) 
  : id_(id),
    electronicsId_(0),
    capId_(capId)
{
  setSize(samples) ;
  setPresamples(presamples) ;
}

void HcalUpgradeDataFrame::setSize(int size) {
  if (size>MAXSAMPLES) size_=MAXSAMPLES;
  else if (size<=0) size_=0;
  else size_=size;
}

void HcalUpgradeDataFrame::setPresamples(int presamples) {
  if (presamples>MAXSAMPLES) presamples_|=MAXSAMPLES&0xF;
  else if (presamples<=0) presamples_=0;
  else presamples_|=presamples&0xF;
}

void HcalUpgradeDataFrame::setReadoutIds(const HcalElectronicsId& eid) {
  electronicsId_=eid;
}

HcalUpgradeQIESample HcalUpgradeDataFrame::operator[](int i) const {
  return HcalUpgradeQIESample(adc(i), (capId_+i)%4, 0, 0);
}

void HcalUpgradeDataFrame::setSample(int iSample,
				     uint16_t adc,
				     uint16_t tdc,
				     bool dv) { 
  if ( iSample < MAXSAMPLES && iSample>=0) { 
    adc_[iSample] = adc&0xFF ; 
    tdc_[iSample] = tdc&0xFFFF ; 
    dv_[iSample] = dv ; 
  }    
}

void HcalUpgradeDataFrame::setZSInfo(bool unsuppressed, bool markAndPass, 
				     uint32_t crossingMask) {
  presamples_&=0x7FC00F0F; // preserve actual presamples and fiber idle offset
  if (markAndPass)  presamples_|=0x10;
  if (unsuppressed) presamples_|=0x20;
  presamples_|=(crossingMask&0x3FF)<<12; 
}



std::ostream& operator<<(std::ostream& s, const HcalUpgradeDataFrame& digi) {
  s << digi.id() << " " << digi.size() << " samples  " << digi.presamples() << " presamples ";
  if (digi.zsUnsuppressed()) s << " zsUS";
  if (digi.zsMarkAndPass())  s << " zsM&P";
  s << std::endl;
  for (int i=0; i<digi.size(); i++) {
    if ( i < digi.presamples() ) s << " (PRE) " ;
    else s << "       " ;
    s << int(digi.capId(i)) << " (capId)  ";
    s << int(digi.adc(i)) << " (adc)  " ;
    s << int(digi.tdc(i)) << " (tdc)  " ;
    if ( digi.valid(i) ) s << " (DV) " ; 
    s << std::endl ;
  }
  return s;
}
  

