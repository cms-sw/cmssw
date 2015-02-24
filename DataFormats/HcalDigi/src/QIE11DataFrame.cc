#include "DataFormats/HcalDigi/interface/QIE11DataFrame.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"

void QIE11DataFrame::setCapid0(int cap0) {
  edm::DataFrame::operator[](0)&=0xFCFF;
  edm::DataFrame::operator[](0)|=((cap0&0x3)<<8);  
}

void QIE11DataFrame::setSample(edm::DataFrame::size_type isample, int adc, int tdc, bool soi) {
  if (isample>=size()) return;
  edm::DataFrame::operator[](isample+1)=(adc&0xFF)|(soi?(0x4000):(0))|((tdc&0x3F)<<8);
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
  if (digi.wasMarkAndPass()) s << " M&P ";
  s << std::endl;
  for (int i=0; i<digi.samples(); i++) {
    s << "  ADC=" << digi[i].adc() << " TDC=" << digi[i].tdc() << " CAPID=" << digi[i].capid();
    if (digi[i].soi()) s << " SOI ";
    s << std::endl;
  }
  return s;
}
