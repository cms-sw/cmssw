#ifndef DataFormats_BTauReco_DeepBoostedJetTagInfo_h
#define DataFormats_BTauReco_DeepBoostedJetTagInfo_h

#include "DataFormats/BTauReco/interface/FeaturesTagInfo.h"
#include "DataFormats/BTauReco/interface/DeepBoostedJetFeatures.h"

namespace reco {

typedef  FeaturesTagInfo<btagbtvdeep::DeepBoostedJetFeatures> DeepBoostedJetTagInfo;

DECLARE_EDM_REFS( DeepBoostedJetTagInfo )

}

#endif // DataFormats_BTauReco_DeepBoostedJetTagInfo_h
