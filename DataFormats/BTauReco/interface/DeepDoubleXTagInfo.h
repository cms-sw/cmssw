#ifndef DataFormats_BTauReco_DeepDoubleXTagInfo_h
#define DataFormats_BTauReco_DeepDoubleXTagInfo_h

#include "DataFormats/BTauReco/interface/FeaturesTagInfo.h"
#include "DataFormats/BTauReco/interface/DeepDoubleXFeatures.h"

namespace reco {

typedef  FeaturesTagInfo<btagbtvdeep::DeepDoubleXFeatures> DeepDoubleXTagInfo;

DECLARE_EDM_REFS( DeepDoubleXTagInfo )

}

#endif // DataFormats_BTauReco_DeepDoubleXTagInfo_h
