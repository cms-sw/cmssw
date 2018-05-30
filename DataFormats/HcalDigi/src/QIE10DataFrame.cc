#include "DataFormats/HcalDigi/interface/QIE10DataFrame.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"

std::ostream& operator<<(std::ostream& s, const QIE10DataFrame& digi) {
  if (digi.detid().det()==DetId::Hcal) {
    s << HcalGenericDetId(digi.detid());
  } else {
    s << "DetId(" << digi.detid().rawId() << ")";    
  }
  s << " " << digi.samples() << " samples";
  if (digi.linkError()) s << " LinkError ";
  if (digi.zsMarkAndPass()) s << " MaP ";
  s << std::endl;
  for (int i=0; i<digi.samples(); i++) {
    QIE10DataFrame::Sample sam = digi[i];
    s << "  ADC=" << sam.adc() << " TDC(LE)=" << sam.le_tdc() << " TDC(TE)=" << sam.te_tdc() << " CAPID=" << sam.capid();
    if (sam.soi()) s << " SOI ";
    if (!sam.ok()) s << " !OK ";
    s << std::endl;
  }
  return s;
}
