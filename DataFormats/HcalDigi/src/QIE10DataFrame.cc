#include "DataFormats/HcalDigi/interface/QIE10DataFrame.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"

void QIE10DataFrame::setSample(edm::DataFrame::size_type isample, int adc, int le_tdc, int fe_tdc, int capid, bool soi, bool ok) {
  if (isample>=size()) return;
  edm::DataFrame::operator[](isample*WORDS_PER_SAMPLE+HEADER_WORDS)=(adc&Sample::MASK_ADC)|(soi?(Sample::MASK_SOI):(0))|(ok?(Sample::MASK_OK):(0));
  edm::DataFrame::operator[](isample*WORDS_PER_SAMPLE+HEADER_WORDS+1)=(le_tdc&Sample::MASK_LE_TDC)|((fe_tdc&Sample::MASK_TE_TDC)<<Sample::OFFSET_TE_TDC)|((capid&Sample::MASK_CAPID)<<Sample::OFFSET_CAPID)|0x4000; // 0x4000 marks this as second word of a pair
}

void QIE10DataFrame::setFlags(uint16_t v) {
  edm::DataFrame::operator[](size()-1)=v;
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
