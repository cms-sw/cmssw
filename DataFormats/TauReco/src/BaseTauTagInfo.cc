#include "DataFormats/TauReco/interface/BaseTauTagInfo.h"

BaseTauTagInfo::BaseTauTagInfo(){}

const TrackRefVector& BaseTauTagInfo::Tracks()const{return Tracks_;}
void BaseTauTagInfo::setTracks(const TrackRefVector x){Tracks_=x;}
