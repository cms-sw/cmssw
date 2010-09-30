#include "RecoLocalTracker/SiStripZeroSuppression/interface/FlatAPVRestorer.h"

#include <cmath>
#include <iostream>

int16_t FlatAPVRestorer::inspect( const uint32_t& detId,std::vector<int16_t>& digis) {return inspect_(detId,digis);}
int16_t FlatAPVRestorer::inspect( const uint32_t& detId,std::vector<float>& digis) {return inspect_(detId,digis);}

void FlatAPVRestorer::restore( std::vector<int16_t>& digis) {restore_(digis);}
void FlatAPVRestorer::restore( std::vector<float>& digis) {restore_(digis);}

void FlatAPVRestorer::init(const edm::EventSetup& es){
  uint32_t q_cache_id = es.get<SiStripQualityRcd>().cacheIdentifier();

  if(q_cache_id != quality_cache_id) {
    es.get<SiStripQualityRcd>().get( qualityHandle );
    quality_cache_id = q_cache_id;
  }
}


template<typename T>
inline
int16_t FlatAPVRestorer::
inspect_(const uint32_t& detId,std::vector<T>& digis){

  SiStripQuality::Range detQualityRange = qualityHandle->getRange(detId);

  typename std::vector<T>::iterator fs;

  apvFlags.clear();
  int16_t nAPVflagged = 0;

  for( uint16_t APV=0; APV< digis.size()/128; ++APV)
  {
    int zeroCount = 0, qualityCount = 0; 
    for (uint16_t istrip=APV*128; istrip<(APV+1)*128; ++istrip)
    {
      fs = digis.begin() + istrip;
      if ( !qualityHandle->IsStripBad(detQualityRange,istrip) )
      {
        qualityCount++; 
        if ( (int) *fs < 1 ) zeroCount++;
      }
    }

    if( zeroCount > restoreThreshold_ * qualityCount ) {
      apvFlags.push_back( true );
      nAPVflagged++;
    } else {
      apvFlags.push_back( false );
    }

  }

  return nAPVflagged;

}

template<typename T>
inline
void FlatAPVRestorer::
restore_( std::vector<T>& digis ){

  typename std::vector<T>::iterator  
  strip( digis.begin() ), 
  endAPV;

  for( uint16_t APV=0; APV< digis.size()/128; ++APV)
  {
    endAPV = digis.begin() + (APV+1)*128;
    if ( *( apvFlags.begin() + APV ) )
    {
      //std::cout << "RESTORING:" << std::endl;
      int counter = 0;
      while (strip < endAPV) {
        *strip = static_cast<T>(150);
        if (counter == 0) *strip = static_cast<T>(0);
        if (counter == 127) *strip = static_cast<T>(0);
        counter++;
        strip++;
      }
    }
  }
}
