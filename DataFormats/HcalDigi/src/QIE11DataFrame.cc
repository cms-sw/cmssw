#include "DataFormats/HcalDigi/interface/QIE11DataFrame.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"

void QIE11DataFrame::setCapid0(int cap0) {
  m_data[0]&=0xFCFF; // inversion of the capid0 mask
  m_data[0]|=((cap0&Sample::MASK_CAPID)<<Sample::OFFSET_CAPID);  
}

void QIE11DataFrame::setFlags(uint16_t v) {
  m_data[size()-1]=v;
}

void QIE11DataFrame::copyContent(const QIE11DataFrame& digi) {
  for (edm::DataFrame::size_type i=0; i<size() && i<digi.size();i++){
    Sample sam = digi[i];
	setSample(i,sam.adc(),sam.tdc(),sam.soi());
  }
}

int QIE11DataFrame::presamples() const {
  for (int i=0; i<samples(); i++) {
    if ((*this)[i].soi()) return i;
  }
  return -1;
}

void QIE11DataFrame::setSample(edm::DataFrame::size_type isample, int adc, int tdc, bool soi) {
  if (isample>=size()) return;
  m_data[isample+1]=(adc&Sample::MASK_ADC)|(soi?(Sample::MASK_SOI):(0))|((tdc&Sample::MASK_TDC)<<Sample::OFFSET_TDC);
}

std::ostream& operator<<(std::ostream& s, const QIE11DataFrame& digi) {
  if (digi.detid().det()==DetId::Hcal) {
    s << HcalGenericDetId(digi.detid());
  } else {
    s << "DetId(" << digi.detid().rawId() << ")";    
  }
  s << " " << digi.samples() << " samples";
  if (digi.linkError()) s << " LinkError ";
  if (digi.capidError()) s << " CapIdError ";
  if (digi.zsMarkAndPass()) s << " M&P ";
  s << std::endl;
  for (int i=0; i<digi.samples(); i++) {
    QIE11DataFrame::Sample sam = digi[i];
    s << "  ADC=" << sam.adc() << " TDC=" << sam.tdc() << " CAPID=" << sam.capid();
    if (sam.soi()) s << " SOI ";
    s << std::endl;
  }
  return s;
}
