#ifndef DataFormats_BTauReco_DeepDoubleBTagInfo_h
#define DataFormats_BTauReco_DeepDoubleBTagInfo_h

#include "DataFormats/BTauReco/interface/FeaturesTagInfo.h"
#include "DataFormats/BTauReco/interface/DeepDoubleBFeatures.h"

namespace reco {

typedef  FeaturesTagInfo<btagbtvdeep::DeepDoubleBFeatures> DeepDoubleBTagInfo;

DECLARE_EDM_REFS( DeepDoubleBTagInfo )

}

#endif // DataFormats_BTauReco_DeepDoubleBTagInfo_h
