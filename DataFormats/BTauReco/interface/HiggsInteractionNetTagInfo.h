#ifndef DataFormats_BTauReco_HiggsInteractionNetTagInfo_h
#define DataFormats_BTauReco_HiggsInteractionNetTagInfo_h

#include "DataFormats/BTauReco/interface/FeaturesTagInfo.h"
#include "DataFormats/BTauReco/interface/HiggsInteractionNetFeatures.h"

namespace reco {

  typedef FeaturesTagInfo<btagbtvdeep::HiggsInteractionNetFeatures> HiggsInteractionNetTagInfo;

  DECLARE_EDM_REFS(HiggsInteractionNetTagInfo)

}  // namespace reco

#endif  // DataFormats_BTauReco_HiggsInteractionNetTagInfo_h
