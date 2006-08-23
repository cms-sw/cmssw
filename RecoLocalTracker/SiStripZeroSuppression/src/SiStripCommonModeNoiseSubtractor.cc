#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripCommonModeNoiseSubtractor.h"
#include <iostream>

void SiStripCommonModeNoiseSubtractor::subtract(std::vector<int16_t>& digis){
  
  LogDebug("SiStripZeroSuppression") << "[SiStripCommonModeNoiseSubtractor::subtract] CMNSubMode " << CMNSubMode << " digis.size()= " << digis.size();

  if (CMNSubMode == "Median") {
    std::vector<int16_t> APVdigis;
    std::vector<int16_t>::iterator fs;
    std::vector<int16_t>::iterator ls;
    float CM;
    APVdigis.reserve(128);
    int nAPV = digis.size()/128;
    LogDebug("SiStripZeroSuppression") << "[SiStripCommonModeNoiseSubtractor::subtract] number of apvs: nAPV= " << nAPV;
    for (int iAPV=0; iAPV<nAPV; iAPV++){
      APVdigis.clear(); //added verify
      fs = digis.begin()+iAPV*128;
      ls = digis.begin()+(iAPV+1)*128;
      APVdigis.insert(APVdigis.end(), fs, ls );
      std::sort(APVdigis.begin(),APVdigis.end());
      CM = (APVdigis[63]+APVdigis[64])/2.;
#ifdef DEBUG
      std::cout << "[SiStripCommonModeNoiseSubtractor::subtract] iApv= " <<iAPV << " CM= " << CM << std::endl;
#endif
      while (fs < ls) {
	*fs = (short) (*fs-CM);
#ifdef DEBUG
	std::cout << "[SiStripCommonModeNoiseSubtractor::subtract] adc CM subtr " << *fs << std::endl;
#endif
	fs++;
      }
    }
  } else {
    edm::LogError("SiStripZeroSuppression") << " Common Mode subtraction mode " <<  CMNSubMode << " not defined"<< std::endl;
    edm::LogError("SiStripZeroSuppression") << " Common Mode subtraction not done"<< std::endl;
  }
}
