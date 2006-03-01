#ifndef BTauReco_TrackTagInfo_h
#define BTauReco_TrackTagInfo_h
//
// $Id: TrackTagInfo.h,v 1.1 2006/02/28 18:18:35 vos Exp $
//
// tagging algorithm specific information for RECO tracks
//
// Author: Marcel Vos 
//

#include "DataFormats/BTauReco/interface/TrackTagInfoFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

namespace reco {
  class TrackTagInfo {
  public:

    TrackTagInfo() { }
    TrackTagInfo(TrackRef track, float probability) : track_ (track), probability_ (probability)  { }
    
    TrackRef track() const { return track_; }
    float probability() const { return probability_;}
    float du() { return 3;}
  private:
    TrackRef track_;

    float probability_;





  };

}

#endif
