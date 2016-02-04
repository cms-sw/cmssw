#include "RecoLocalTracker/SiStripZeroSuppression/interface/FastLinearCMNSubtractor.h"

void FastLinearCMNSubtractor::subtract(const uint32_t& detId,std::vector<int16_t>& digis){ subtract_(detId,digis);}
void FastLinearCMNSubtractor::subtract(const uint32_t& detId,std::vector<float>& digis){ subtract_(detId,digis);}

template<typename T>
inline
void FastLinearCMNSubtractor::
subtract_(const uint32_t& detId,std::vector<T>& digis){

  std::vector<T> tmp;  tmp.reserve(128);
  typename std::vector<T>::iterator 
    strip( digis.begin() ), 
    end(   digis.end()   ),
    endAPV, high, low;

  while( strip < end ) {
    endAPV = strip+128; tmp.clear();
    tmp.insert(tmp.end(),strip,endAPV);
    const float offset = median(tmp);
    
    low = strip;   high = strip+64;   tmp.clear(); 
    while( high < endAPV) tmp.push_back( *high++ - *low++ );
    const float slope = median(tmp)/64.;

    while (strip < endAPV) {
      *strip = static_cast<T>( *strip - (offset + slope*(65 - (endAPV-strip) ) ) );
      strip++;
    }

  }
}

// Details on http://abbaneo.web.cern.ch/abbaneo/cmode/cm.html

