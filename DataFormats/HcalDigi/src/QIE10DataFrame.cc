#include "DataFormats/HcalDigi/interface/QIE10DataFrame.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"

void QIE10DataFrame::setSample(edm::DataFrame::size_type isample, int adc, int le_tdc, int te_tdc, int capid, bool soi, bool ok) {
  if (isample>=size()) return;
  m_data[isample*WORDS_PER_SAMPLE+HEADER_WORDS]=(adc&Sample::MASK_ADC)|(soi?(Sample::MASK_SOI):(0))|(ok?(Sample::MASK_OK):(0));
  m_data[isample*WORDS_PER_SAMPLE+HEADER_WORDS+1]=(le_tdc&Sample::MASK_LE_TDC)|((te_tdc&Sample::MASK_TE_TDC)<<Sample::OFFSET_TE_TDC)|((capid&Sample::MASK_CAPID)<<Sample::OFFSET_CAPID)|0x4000; // 0x4000 marks this as second word of a pair
}

void QIE10DataFrame::setFlags(uint16_t v) {
  m_data[size()-1]=v;
}

void QIE10DataFrame::copyContent(const QIE10DataFrame& digi) {
  for (edm::DataFrame::size_type i=0; i<size() && i<digi.size();i++){
    Sample sam = digi[i];
    setSample(i, sam.adc(), sam.le_tdc(), sam.te_tdc(), sam.capid(), sam.soi(), sam.ok());
  }
}

int QIE10DataFrame::presamples() const {
  for (int i=0; i<samples(); i++) {
    if ((*this)[i].soi()) return i;
  }
  return -1;
}

void QIE10DataFrame::setZSInfo(bool markAndPass){
	if(markAndPass) m_data[0] |= MASK_MARKPASS;
}

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
