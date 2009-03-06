#ifndef BaseDeDxEstimator_h
#define BaseDeDxEstimator_h
#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"

class BaseDeDxEstimator
{
public: 
 virtual float dedx(const reco::TrackDeDxHits & trackWithHits) = 0;



};

#endif
