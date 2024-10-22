#include "DataFormats/HcalDigi/interface/QIE11DataFrame.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"

std::ostream& operator<<(std::ostream& s, const QIE11DataFrame& digi) {
  if (digi.detid().det() == DetId::Hcal) {
    s << "DetID=" << HcalGenericDetId(digi.detid()) << " flavor=" << digi.flavor();
  } else {
    s << "DetId(" << digi.detid().rawId() << ")";
  }
  s << " " << digi.samples() << " samples";
  if (digi.linkError())
    s << " LinkError ";
  if (digi.capidError())
    s << " CapIdError ";
  if (digi.zsMarkAndPass())
    s << " M&P ";
  s << std::endl;
  for (int i = 0; i < digi.samples(); i++) {
    QIE11DataFrame::Sample sam = digi[i];
    s << "  ADC=" << sam.adc() << " TDC=" << sam.tdc() << " CAPID=" << sam.capid();
    if (sam.soi())
      s << " SOI ";
    s << std::endl;
  }
  return s;
}
