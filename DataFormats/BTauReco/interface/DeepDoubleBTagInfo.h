#ifndef DataFormats_BTauReco_DeepDoubleBTagInfo_h
#define DataFormats_BTauReco_DeepDoubleBTagInfo_h

#include "DataFormats/BTauReco/interface/FeaturesTagInfo.h"
#include "DataFormats/BTauReco/interface/DeepDoubleXFeatures.h"

//keeping for posterity
namespace reco {

typedef  FeaturesTagInfo<btagbtvdeep::DeepDoubleXFeatures> DeepDoubleBTagInfo;

DECLARE_EDM_REFS( DeepDoubleBTagInfo )

}

#endif // DataFormats_BTauReco_DeepDoubleBTagInfo_h
