#ifndef TrackReco_HitPatternFiller_h
#define TrackReco_HitPatternFiller_h
/* \class HitPatternFiller 
 *
 * Fill HitPattern from a collection of hits
 *
 * \author Luca Lista, INFN
 *
 * $Id$
 *
 */
#include "DataFormats/Common/interface/OwnVector.h"

class TrackingRecHit;
namespace reco { class HitPattern; }

struct HitPatternFiller {
  typedef edm::OwnVector<TrackingRecHit>::const_iterator const_iterator;

  template<typename I>
  static void fill( I begin, I end, reco::HitPattern & hitPattern );
private:
  static void fill( const TrackingRecHit &, int counter, reco::HitPattern & );
};

template<typename I>
void HitPatternFiller::fill( I begin, I end, reco::HitPattern & hitPattern ) {
  hitPattern.clear();
  int counter = 0;
  for ( I hit = begin; 
	hit != end && counter < reco::HitPattern::numberOfPatterns;
	hit++, counter++ ) {
    fill( * hit, counter, hitPattern );
  }
}

#endif
