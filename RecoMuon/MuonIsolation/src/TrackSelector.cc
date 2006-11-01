#include "TrackSelector.h"

using namespace muonisolation;
using namespace reco;
TrackSelector::TrackSelector(const Range & z, float r, const Direction & dir, float drMax)
  : theZ(z), theR(Range(0.,r)), theDir(dir), theDR_Max(drMax)
{ } 

TrackCollection TrackSelector::operator()(const TrackCollection & tracks) const
{
  TrackCollection result;
  for (TrackCollection::const_iterator it = tracks.begin(); it != tracks.end(); it++) {
    if ( !theZ.inside( (*it).vz() ) ) continue; 
    if ( !theR.inside( fabs((*it).d0()) ) ) continue;
    if ( theDir.deltaR( Direction(it->eta(), it->phi0()) ) > theDR_Max ) continue;
    result.push_back(*it);
  } 
  return result;
}
