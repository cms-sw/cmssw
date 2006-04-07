#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripCommonModeNoiseSubtractor.h"

void SiStripCommonModeNoiseSubtractor::subtract(std::vector<int16_t>& digis){

  if (CMNSubMode == "Median") {
    std::vector<int16_t> APVdigis;
    std::vector<int16_t>::iterator fs;
    std::vector<int16_t>::iterator ls;
    float CM;
    APVdigis.reserve(128);
    int nAPV = digis.size()%128;
    for (int iAPV=0; iAPV<nAPV; iAPV++){
      fs = digis.begin()+iAPV*128;
      ls = digis.begin()+(iAPV+1)*128;
      APVdigis.insert(APVdigis.end(), fs, ls );
      std::sort(APVdigis.begin(),APVdigis.end());
      CM = (APVdigis[63]+APVdigis[64])/2.;
      while (fs < ls) {
	*fs = (short) (*fs-CM);
	fs++;
      }
    }
  } else {
    edm::LogError("SiStrip") << " Common Mode subtraction mode " <<  CMNSubMode << " not defined"<< std::endl;
    edm::LogError("SiStrip") << " Common Mode subtraction not done"<< std::endl;
  }
}
