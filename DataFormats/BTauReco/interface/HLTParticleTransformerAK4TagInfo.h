#ifndef DataFormats_BTauReco_HLTParticleTransformerAK4TagInfo_h
#define DataFormats_BTauReco_HLTParticleTransformerAK4TagInfo_h

#include "DataFormats/BTauReco/interface/HLTParticleTransformerAK4Features.h"
#include "DataFormats/BTauReco/interface/FeaturesTagInfo.h"

namespace reco {
  typedef FeaturesTagInfo<btagbtvdeep::HLTParticleTransformerAK4Features> HLTParticleTransformerAK4TagInfo;
  DECLARE_EDM_REFS(HLTParticleTransformerAK4TagInfo)
}  // namespace reco

#endif  // DataFormats_BTauReco_HLTParticleTransformerAK4TagInfo_h
