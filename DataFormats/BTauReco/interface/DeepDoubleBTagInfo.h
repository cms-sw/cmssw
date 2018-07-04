#ifndef DataFormats_BTauReco_DeepDoubleBTagInfo_h
#define DataFormats_BTauReco_DeepDoubleBTagInfo_h

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DataFormats/BTauReco/interface/DeepFlavourTagInfo.h"
#include "DataFormats/BTauReco/interface/DeepDoubleBFeatures.h"

#include "DataFormats/PatCandidates/interface/Jet.h"

namespace reco {

typedef  FeaturesTagInfo<btagbtvdeep::DeepDoubleBFeatures> DeepDoubleBTagInfo;

DECLARE_EDM_REFS( DeepDoubleBTagInfo )

}

#endif // DataFormats_BTauReco_DeepDoubleBTagInfo_h
