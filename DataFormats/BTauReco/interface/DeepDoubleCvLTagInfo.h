#ifndef DataFormats_BTauReco_DeepDoubleCvLTagInfo_h
#define DataFormats_BTauReco_DeepDoubleCvLTagInfo_h

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DataFormats/BTauReco/interface/DeepFlavourTagInfo.h"
#include "DataFormats/BTauReco/interface/DeepDoubleCvLFeatures.h"

#include "DataFormats/PatCandidates/interface/Jet.h"

namespace reco {

typedef  FeaturesTagInfo<btagbtvdeep::DeepDoubleCvLFeatures> DeepDoubleCvLTagInfo;

DECLARE_EDM_REFS( DeepDoubleCvLTagInfo )

}

#endif // DataFormats_BTauReco_DeepDoubleCvLTagInfo_h
