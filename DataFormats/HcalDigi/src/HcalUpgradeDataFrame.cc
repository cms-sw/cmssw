#include "DataFormats/HcalDigi/interface/HcalUpgradeDataFrame.h"


HcalUpgradeDataFrame::HcalUpgradeDataFrame() : id_(0), 
                                     capId_(0),
                                     size_(0),
                                     presamples_(0)
{
}

HcalUpgradeDataFrame::HcalUpgradeDataFrame(HcalDetId id, int capId, int samples, int presamples) : id_(id),
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
  if (presamples>MAXSAMPLES) presamples_=MAXSAMPLES;
  else if (presamples<=0) presamples_=0;
  else presamples_=presamples;
}

void HcalUpgradeDataFrame::setSample(int relativeSample,
                                const uint16_t adc,
                                const uint8_t tdc,
                                const bool dv) { 

    int linSample = presamples_ + relativeSample ;
    if ( linSample < MAXSAMPLES && linSample>=0) { 
        adc_[linSample] = adc&0xFF ; 
        tdc_[linSample] = tdc&0x1F ; 
        dv_[linSample] = dv ; 
    }    
}

std::ostream& operator<<(std::ostream& s, const HcalUpgradeDataFrame& digi) {
    s << digi.id() << " " << digi.size() << " samples  " << digi.presamples() << " presamples ";
    for (int i=0; i<digi.size(); i++) {
        int relSample = i - digi.presamples() ; 
        if ( relSample < 0 ) s << " (PRE) " ;
        else s << "       " ;
        s << int(digi.adc(relSample)) << " (adc)  " ;
	s << int(digi.tdc(relSample)) << " (tdc)  " ;
        if ( digi.valid(relSample) ) s << " (DV) " ; 
        s << std::endl ;
    }
    return s;
}
  

