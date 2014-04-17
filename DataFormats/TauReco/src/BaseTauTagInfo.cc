#include "DataFormats/TauReco/interface/BaseTauTagInfo.h"
using namespace reco;
BaseTauTagInfo::BaseTauTagInfo(){}

const reco::TrackRefVector& BaseTauTagInfo::Tracks()const{return Tracks_;}
void reco::BaseTauTagInfo::setTracks(const reco::TrackRefVector& x){Tracks_=x;}
