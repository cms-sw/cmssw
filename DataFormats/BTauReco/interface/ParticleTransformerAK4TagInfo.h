#ifndef DataFormats_BTauReco_ParticleTransformerAK4TagInfo_h
#define DataFormats_BTauReco_ParticleTransformerAK4TagInfo_h

#include "DataFormats/BTauReco/interface/ParticleTransformerAK4Features.h"
#include "DataFormats/BTauReco/interface/FeaturesTagInfo.h"

namespace reco {

  typedef FeaturesTagInfo<btagbtvdeep::ParticleTransformerAK4Features> ParticleTransformerAK4TagInfo;

  DECLARE_EDM_REFS(ParticleTransformerAK4TagInfo)

}  // namespace reco

#endif  // DataFormats_BTauReco_ParticleTransformerAK4TagInfo_h
