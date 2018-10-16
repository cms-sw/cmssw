#ifndef DataFormats_BTauReco_DeepDoubleCvBTagInfo_h
#define DataFormats_BTauReco_DeepDoubleCvBTagInfo_h

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DataFormats/BTauReco/interface/DeepFlavourTagInfo.h"
#include "DataFormats/BTauReco/interface/DeepDoubleCvBFeatures.h"

#include "DataFormats/PatCandidates/interface/Jet.h"

namespace reco {

typedef  FeaturesTagInfo<btagbtvdeep::DeepDoubleCvBFeatures> DeepDoubleCvBTagInfo;

DECLARE_EDM_REFS( DeepDoubleCvBTagInfo )

}

#endif // DataFormats_BTauReco_DeepDoubleCvBTagInfo_h
