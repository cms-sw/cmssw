#include "DataFormats/TrackReco/interface/HitPattern.h"

using namespace reco;

void HitPattern::clear() {
  for ( int i = 0 ; i < patternSize ; i ++ ) 
    hitPattern_[ i ] = 0;
}

bool HitPattern::hasValidHitInFirstPixelBarrel() const {
  for ( int i = 0 ; i < ( patternSize * 32 ) / hitSize ; i++ ) {
    uint32_t pattern = getHitPattern( i );
    if ( validHitFilter( pattern ) && trackerHitFilter( pattern ) ) {
      uint32_t substructure = ( pattern >> substrOffset ) & substrMask + 1;
      if ( substructure == PixelSubdetector::PixelBarrel ) {
	int layer = ( pattern >> layerOffset )  & layerMask + 1;
	if ( layer == 1 ) return true;
      }
    }
    
  }
  return false;
}

uint32_t HitPattern::getHitPattern( int position ) const {
  int offset = position * hitSize;
  uint32_t pattern = 0; 
  for ( int i = 0; i < hitSize; i ++ ) {
    int pos = offset + i;
    uint32_t word = hitPattern_[ pos / 32 ];
    uint32_t bit = ( word >> ( pos % 32 ) ) & 0x1;
    pattern += bit << i;
  }
  return pattern;
}

void HitPattern::setHitPattern( int position, uint32_t pattern ) {
  int offset = position * hitSize;
  for ( int i = 0; i < hitSize; i ++ ) {
    int pos = offset + i;
    uint32_t bit = ( pattern >> i ) & 0x1;
    hitPattern_[ pos / 32 ] += bit << ( (offset + i) % 32 ); 
  }
}

bool HitPattern::validHitFilter(uint32_t pattern) const {
  return ( pattern >> validOffset ) & validMask;
}

bool HitPattern::trackerHitFilter(uint32_t pattern) const {
  return DetId::Detector( ( pattern >> subDetectorOffset ) & subDetectorMask ) == DetId::Tracker;
}

bool HitPattern::muonHitFilter(uint32_t pattern) const {
  return DetId::Detector( ( pattern >> subDetectorOffset ) & subDetectorMask ) == DetId::Muon;
}

bool HitPattern::pixelHitFilter(uint32_t pattern) const { 
  if ( ! trackerHitFilter( pattern ) ) return false;
  uint32_t substructure = ( pattern >> substrOffset ) & substrMask;
  return 
    substructure == PixelSubdetector::PixelBarrel || 
    substructure == PixelSubdetector::PixelEndcap;
}

unsigned int HitPattern::numberOfValidHits() const {
  unsigned int count = 0;
  for ( int i = 0 ; i < numberOfPatterns ; i ++ ) {
    uint32_t pattern = getHitPattern( i );
    if ( validHitFilter( pattern ) ) count ++;
  }
  return count;
}

unsigned int HitPattern::numberOfLostHits() const {
  unsigned int count = 0;
  for ( int i = 0 ; i < numberOfPatterns ; i ++ ) {
    uint32_t pattern = getHitPattern(i);
    if ( muonHitFilter( pattern ) || trackerHitFilter( pattern ) ) {
      if ( ! validHitFilter( pattern ) ) count++;
    }
  }
  return count;
}

      
unsigned int HitPattern::numberOfValidMuonHits() const {
  unsigned int count = 0;
  for ( int i = 0 ; i < numberOfPatterns ; i ++ ) {
    uint32_t pattern = getHitPattern( i );
    if ( validHitFilter( pattern ) && muonHitFilter( pattern ) ) count ++;
  }
  
  return count;
}

unsigned int HitPattern::numberOfLostMuonHits() const {
  unsigned int count = 0;
  for ( int i = 0 ; i < numberOfPatterns ; i ++ ) {
    uint32_t pattern = getHitPattern( i );
    if ( ! validHitFilter(pattern) && muonHitFilter( pattern ) ) count ++;
    
  }
  return count;
}

unsigned int HitPattern::numberOfValidTrackerHits() const {
  unsigned int count = 0;
  for ( int i = 0 ; i < numberOfPatterns ; i ++ ) {
    uint32_t pattern = getHitPattern( i );
    if ( validHitFilter( pattern ) && trackerHitFilter( pattern ) ) count ++;
  }
  return count;
}

unsigned int HitPattern::numberOfLostTrackerHits() {
  unsigned int count = 0;
  for (int i = 0 ; i < numberOfPatterns ; i++) {
    uint32_t pattern = getHitPattern(i);
    if (!validHitFilter(pattern) && trackerHitFilter(pattern)) count++;
    
  }
  return count;
}

unsigned int HitPattern::numberOfValidPixelHits() const {
  unsigned int count = 0;
  for ( int i = 0 ; i < numberOfPatterns ; i ++ ) {
    uint32_t pattern = getHitPattern( i );
    if ( validHitFilter( pattern ) && pixelHitFilter( pattern ) ) count ++;
    
  }
  return count;
}

unsigned int HitPattern::numberOfLostPixelHits() const {
  unsigned int count = 0;
  for ( int i = 0 ; i < numberOfPatterns ; i ++ ) {
    uint32_t pattern = getHitPattern( i );
    if ( ! validHitFilter( pattern ) && pixelHitFilter( pattern ) ) count ++;
  }
  return count;
}

