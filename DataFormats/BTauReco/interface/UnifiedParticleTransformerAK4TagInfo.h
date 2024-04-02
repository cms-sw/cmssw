#ifndef DataFormats_BTauReco_UnifiedParticleTransformerAK4TagInfo_h
#define DataFormats_BTauReco_UnifiedParticleTransformerAK4TagInfo_h

#include "DataFormats/BTauReco/interface/UnifiedParticleTransformerAK4Features.h"
#include "DataFormats/BTauReco/interface/FeaturesTagInfo.h"

namespace reco {

  typedef FeaturesTagInfo<btagbtvdeep::UnifiedParticleTransformerAK4Features> UnifiedParticleTransformerAK4TagInfo;

  DECLARE_EDM_REFS(UnifiedParticleTransformerAK4TagInfo)

}  // namespace reco

#endif  // DataFormats_BTauReco_UnifiedParticleTransformerAK4TagInfo_h
