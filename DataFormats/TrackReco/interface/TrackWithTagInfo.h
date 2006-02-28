#ifndef BTauReco_TrackWithTagInfo_h
#define BTauReco_TrackWithTagInfo_h
//
// $Id: TrackWithTagInfo.h v1.0 2006/01/23 mvos $
//
// tagging algorithm specific information for RECO tracks
//
// Author: Marcel Vos 
//

#include "DataFormats/BTauReco/interface/TrackWithTagInfoFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

namespace reco {
  class TrackWithTagInfo {
  public:

    TrackWithTagInfo() { }
    TrackWithTagInfo(TrackRef track, float probability) : track_ (track), probability_ (probability)  { }
    
    TrackRef track() const { return track_; }
    float probability() const { return probability_;}
    float du() { return 3;}
  private:
    TrackRef track_;

    float probability_;





  };

}

#endif
