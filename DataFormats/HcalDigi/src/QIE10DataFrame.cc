#include "DataFormats/HcalDigi/interface/QIE10DataFrame.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"

void QIE10DataFrame::setSample(edm::DataFrame::size_type isample, int adc, int le_tdc, int fe_tdc, int capid, bool soi, bool ok) {
  if (isample>=size()) return;
  edm::DataFrame::operator[](isample*2+1)=(adc&0xFF)|(soi?(0x2000):(0))|(ok?(0x1000):(0));
  edm::DataFrame::operator[](isample*2+2)=(le_tdc&0x3F)|((fe_tdc&0x1F)<<6)|((capid&0x3)<<12)|0x4000;
}

std::ostream& operator<<(std::ostream& s, const QIE10DataFrame& digi) {
  if (digi.detid().det()==DetId::Hcal) {
    s << HcalGenericDetId(digi.detid());
  } else {
    s << "DetId(" << digi.detid().rawId() << ")";    
  }
  s << " " << digi.samples() << " samples";
  if (digi.linkError()) s << " LinkError ";
  if (digi.wasMarkAndPass()) s << " MaP ";
  s << std::endl;
  for (int i=0; i<digi.samples(); i++) {
    s << "  ADC=" << digi[i].adc() << " TDC(LE)=" << digi[i].le_tdc() << " TDC(TE)=" << digi[i].te_tdc() << " CAPID=" << digi[i].capid();
    if (digi[i].soi()) s << " SOI ";
    if (!digi[i].ok()) s << " !OK ";
    s << std::endl;
  }
  return s;
}
