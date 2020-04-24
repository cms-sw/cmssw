#ifndef DataFormats_BTauReco_DeepFlavourTagInfo_h
#define DataFormats_BTauReco_DeepFlavourTagInfo_h

#include "DataFormats/BTauReco/interface/DeepFlavourFeatures.h"
#include "DataFormats/BTauReco/interface/FeaturesTagInfo.h"

namespace reco {

typedef  FeaturesTagInfo<btagbtvdeep::DeepFlavourFeatures> DeepFlavourTagInfo;

DECLARE_EDM_REFS( DeepFlavourTagInfo )

}

#endif // DataFormats_BTauReco_DeepFlavourTagInfo_h
