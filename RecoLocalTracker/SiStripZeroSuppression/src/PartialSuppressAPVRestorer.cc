#include "RecoLocalTracker/SiStripZeroSuppression/interface/PartialSuppressAPVRestorer.h"

#include <cmath>
#include <iostream>

int16_t PartialSuppressAPVRestorer::inspect( const uint32_t& detId,std::vector<int16_t>& digis) {return inspect_(detId,digis);}
int16_t PartialSuppressAPVRestorer::inspect( const uint32_t& detId,std::vector<float>& digis) {return inspect_(detId,digis);}

void PartialSuppressAPVRestorer::restore( std::vector<int16_t>& digis) {restore_(digis);}
void PartialSuppressAPVRestorer::restore( std::vector<float>& digis) {restore_(digis);}

void PartialSuppressAPVRestorer::init(const edm::EventSetup& es){
  uint32_t q_cache_id = es.get<SiStripQualityRcd>().cacheIdentifier();

  if(q_cache_id != quality_cache_id) {
    es.get<SiStripQualityRcd>().get( qualityHandle );
    quality_cache_id = q_cache_id;
  }
}

template<typename T>
inline
int16_t PartialSuppressAPVRestorer::
inspect_(const uint32_t& detId,std::vector<T>& digis){

  SiStripQuality::Range detQualityRange = qualityHandle->getRange(detId);

  typename std::vector<T>::iterator fs;

  apvFlags.clear();
  int16_t nAPVflagged=0;

  int devCount = 0, qualityCount = 0, minstrip = 0; 
  for( uint16_t APV=0; APV< digis.size()/128; ++APV)
  {
  
    for (uint16_t istrip=APV*128; istrip<(APV+1)*128; ++istrip)
    {
      fs = digis.begin() + istrip;
      if ( !qualityHandle->IsStripBad(detQualityRange,istrip) )
      {
        qualityCount++; 
        if ( std::abs((int) *fs - 128) > deviation_ ) devCount++;
	minstrip = std::min((int) *fs, minstrip);
      }
    }

    if( devCount > fraction_ * qualityCount ) {
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
void PartialSuppressAPVRestorer::
restore_( std::vector<T>& digis ){

  return;

}
