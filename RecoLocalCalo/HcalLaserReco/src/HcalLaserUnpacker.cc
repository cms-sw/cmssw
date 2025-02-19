#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/HcalDigi/interface/HcalLaserDigi.h"
#include "RecoLocalCalo/HcalLaserReco/src/HcalLaserUnpacker.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <ostream>

HcalLaserUnpacker::HcalLaserUnpacker(){}

struct CombinedTDCQDCDataFormat {
  unsigned int cdfHeader0,cdfHeader1,cdfHeader2,cdfHeader3;
  unsigned int n_qdc_hits; // Count of QDC channels
  unsigned int n_tdc_hits; // upper/lower TDC counts    
  unsigned short qdc_values[4];
};

void HcalLaserUnpacker::unpack(const FEDRawData& raw,
			       HcalLaserDigi& digi) const {

  if (raw.size()<3*8) {
    throw cms::Exception("Missing Data") << "No data in the block";
  }

  const CombinedTDCQDCDataFormat* qdctdc=(const CombinedTDCQDCDataFormat*)raw.data();

  // first, we do the QADC
  std::vector<uint16_t> qadcvals;
  for (unsigned int i=0;i<qdctdc->n_qdc_hits;i++) {
    qadcvals.push_back(qdctdc->qdc_values[i]&0xFFF);
  }
  digi.setQADC(qadcvals);

  // next, we do the TDC
  const unsigned int* hitbase=(&(qdctdc->n_tdc_hits))+1; // base is one beyond 
  unsigned int totalhits=0;

  hitbase+=qdctdc->n_qdc_hits/2; // two unsigned short per unsigned long
  totalhits=qdctdc->n_tdc_hits&0xFFFF; // mask off high bits

  for (unsigned int i=0; i<totalhits; i++) {
    int channel=(hitbase[i]&0x7FC00000)>>22; // hardcode channel assignment
    int time=(hitbase[i]&0xFFFFF);
    if (channel==0 && time==0 && i==(totalhits-1)) continue; // ignore "filler" hit
    digi.addTDCHit(channel,time);
  }  
}
