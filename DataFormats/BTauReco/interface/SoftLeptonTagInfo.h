#ifndef DataFormats_BTauReco_SoftLeptonTagInfo_h
#define DataFormats_BTauReco_SoftLeptonTagInfo_h

#include <vector>
#include <limits>

#include "DataFormats/BTauReco/interface/RefMacros.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BTauReco/interface/TemplatedSoftLeptonTagInfo.h"

namespace reco {

typedef TemplatedSoftLeptonTagInfo<TrackBaseRef> SoftLeptonTagInfo;

DECLARE_EDM_REFS( SoftLeptonTagInfo )

}

#endif // DataFormats_BTauReco_SoftLeptonTagInfo_h
