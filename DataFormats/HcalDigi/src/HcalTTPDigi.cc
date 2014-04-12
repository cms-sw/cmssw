#include "DataFormats/HcalDigi/interface/HcalTTPDigi.h"

HcalTTPDigi::HcalTTPDigi() : identifier_(0), samples_(0), presamples_(0), fwVersion_(0), algorithm_(0), lPipe_(0) {
    for (int i=0; i<8; i++) {
        algoDepend_[i] = 0 ; triggerOutput_[i] = 0 ;
        for (int j=0; j<5; j++) triggerInputs_[j*8+i] = 0 ;
    }
}

HcalTTPDigi::HcalTTPDigi(int identifier, int samples, int presamples,
                         unsigned int fwVersion, int algorithm, unsigned int lPipe) :
    identifier_(identifier),
    samples_(samples),
    presamples_(presamples),
    fwVersion_(fwVersion),
    algorithm_(algorithm),
    lPipe_(lPipe) {
    for (int i=0; i<8; i++) {
        algoDepend_[i] = 0x0 ; triggerOutput_[i] = 0x0 ;
        for (int j=0; j<5; j++) triggerInputs_[i*5+j] = 0x0 ;
    }
}

void HcalTTPDigi::setSample(int relativeSample,
                            const uint16_t* triggerInputs,
                            const uint32_t algodep,
                            const uint8_t outputTrigger) {

    int linSample=presamples_+relativeSample;
    if (linSample>=0 && linSample<samples_) {
        // Trigger input: 72 bits
        for (int i=0; i<4; i++)
            triggerInputs_[5*linSample+i] = triggerInputs[i] ;
        triggerInputs_[5*linSample+4] = triggerInputs[4]&0xFF ;
        // Algo dependency: 20 bits
        algoDepend_[linSample] = algodep&0xFFFFF ;
        // Trigger output: 4 bits
        triggerOutput_[linSample] = outputTrigger&0xF ;        
    }
}

std::vector<bool> HcalTTPDigi::inputPattern(int relativeSample) const {
  std::vector<bool> retval;
  int linSample=presamples_+relativeSample;
  if (linSample>=0 && linSample<samples_) {
    
    for (int i=0; i<72; i++) {
      int ioff=i/16;
      retval.push_back(triggerInputs_[linSample*5+ioff]&(1<<(i%16)));
    }
  }
  return retval ; 
}

uint8_t HcalTTPDigi::triggerOutput(int relativeSample) const {
  int linSample=presamples_+relativeSample;
  if (linSample>=0 && linSample<samples_) return triggerOutput_[linSample];
  else return 0;
}

uint32_t HcalTTPDigi::algorithmWord(int relativeSample) const {
  int linSample=presamples_+relativeSample;
  if (linSample>=0 && linSample<samples_) return algoDepend_[linSample];
  else return 0;
}

bool HcalTTPDigi::operator==(const HcalTTPDigi& digi) const {

    if (samples_ != digi.size() || presamples_ != digi.presamples()) return false ;
    int relativeSize = digi.size() - digi.presamples() ; 
    for (int i=-this->presamples(); i<relativeSize; i++) {
        if (this->inputPattern(i) != digi.inputPattern(i)) return false ; 
        if (this->algorithmWord(i) != digi.algorithmWord(i)) return false ; 
        if (this->triggerOutput(i) != digi.triggerOutput(i)) return false ; 
    }
    return true ;
}

std::ostream& operator<<(std::ostream& out, const HcalTTPDigi& digi) {

    out << "HcalTTPDigi " << digi.id() 
        << " with " << digi.size() << " samples, "
        << digi.presamples() << " presamples. "
        << std::endl ;
    out << "Firmware version " << digi.fwVersion() << " and flavor/algo " << digi.algorithm() ; 
    out << "; pipeline length " << digi.pipelineLength() << std::endl ;  
    int relativeSize = digi.size() - digi.presamples() ; 
    for (int i=-digi.presamples(); i<relativeSize; i++) {
        for (unsigned int j=digi.inputPattern(i).size(); j>0; j--) {
            if ( !(j%16) ) out << " " ;
            out << digi.inputPattern(i).at(j-1) ;
        }
        if (i < 0) out << " (PRE)" ; // Indicates presamples 
        out << std::endl ; 
        out << "ALGO: " ; 
        for (int j=19; j>=0; j--) out << bool((digi.algorithmWord(i))&(1<<j)) ;

        out << "  TRIG: " ;
        for (int j=3; j>=0; j--) out << bool((digi.triggerOutput(i))&(1<<j)) ; 
        out << std::endl ; 
    }

    return out ; 
}

