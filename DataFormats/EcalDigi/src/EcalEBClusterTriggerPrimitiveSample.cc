#include "DataFormats/EcalDigi/interface/EcalEBClusterTriggerPrimitiveSample.h"



EcalEBClusterTriggerPrimitiveSample::EcalEBClusterTriggerPrimitiveSample() : theSample(0) { }
EcalEBClusterTriggerPrimitiveSample::EcalEBClusterTriggerPrimitiveSample(uint64_t data) : theSample(data) { }

EcalEBClusterTriggerPrimitiveSample::EcalEBClusterTriggerPrimitiveSample(int encodedEt, bool isASpike) { 
  theSample=(encodedEt&0x3FF)| ((isASpike)?(0x400):(0));
}


EcalEBClusterTriggerPrimitiveSample::EcalEBClusterTriggerPrimitiveSample(int encodedEt, bool isASpike, int timing) { 
  theSample=(encodedEt&0x3FF)| ((isASpike)?(0x400):(0)) | timing<<11;
}


EcalEBClusterTriggerPrimitiveSample::EcalEBClusterTriggerPrimitiveSample(int encodedEt) { 
  theSample=encodedEt&0x3FF;
}



std::ostream& operator<<(std::ostream& s, const EcalEBClusterTriggerPrimitiveSample& samp) {
  return s << "ET=" << samp.encodedEt() << ", isASpike=" << samp.l1aSpike()<< " timing= " << samp.time() ; 

}


