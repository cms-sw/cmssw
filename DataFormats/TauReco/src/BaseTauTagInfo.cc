#include "DataFormats/TauReco/interface/BaseTauTagInfo.h"

BaseTauTagInfo::BaseTauTagInfo() : alternatLorentzVect_(0.,0.,0.,0.){}

const TrackRefVector& BaseTauTagInfo::Tracks()const{return Tracks_;}
void BaseTauTagInfo::setTracks(const TrackRefVector x){Tracks_=x;}

const math::XYZTLorentzVector BaseTauTagInfo::alternatLorentzVect()const{return(alternatLorentzVect_);} 
void BaseTauTagInfo::setalternatLorentzVect(math::XYZTLorentzVector x){alternatLorentzVect_=x;}
