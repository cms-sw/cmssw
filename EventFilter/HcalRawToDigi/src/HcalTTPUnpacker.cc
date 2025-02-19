#include "EventFilter/HcalRawToDigi/interface/HcalTTPUnpacker.h"

bool HcalTTPUnpacker::unpack(const HcalHTRData& theData, HcalTTPDigi& theDigi) {

    // Get the base information for the TTP Digi
    int theID    = theData.getSubmodule() ;
    int nSamples = theData.getNDD() ;
    int nPresamples = theData.getNPS() ;
    int algorithm = theData.getFirmwareFlavor()&0x1F ;
    unsigned int fwVersion = theData.getFirmwareRevision() ;
    unsigned int lPipe = theData.getPipelineLength() ; 

    if (nSamples>8) nSamples=8; // protection in case nSamples is too large

    theDigi = HcalTTPDigi(theID,nSamples,nPresamples,fwVersion,algorithm,lPipe) ;

    // Get the data pointers
    const unsigned short *daq_first, *daq_last, *tp_first, *tp_last ;
    theData.dataPointers(&daq_first,&daq_last,&tp_first,&tp_last) ;

    // Each TTP data sample is 96 bits: 72 (input) + 20 (algo) + output (4)
    for (int i=0; i<nSamples; i++) {

        const uint16_t* daq_start = (daq_first+6*i) ;
        if ( daq_start > daq_last ) break ;

        const uint16_t* inputContent = daq_start ; 
        const uint32_t algoDep = (daq_start[4]>>8)|((uint32_t(daq_start[5])&0xFFF)<<8) ; 
        const uint8_t trigOutput = (daq_start[5]>>12)&0xF ;
        
        int relativeSample = i - nPresamples ;
        theDigi.setSample(relativeSample,inputContent,algoDep,trigOutput) ;
    }
    return true;
}

