#include <limits>

#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "RecoBTag/SoftLepton/interface/LeptonTaggerDistance.h"

/// b-tag a jet based on track-to-jet pseudo-angular distance
float LeptonTaggerDistance::discriminator(const TagInfoHelper & tagInfo) const {
  const reco::SoftLeptonTagInfo & info = tagInfo.get<reco::SoftLeptonTagInfo>();
  // if there are any leptons, look for any one within the requested deltaR
  for (unsigned int i = 0; i < info.leptons(); i++) {
    if (info.properties(i).deltaR <= m_maxDistance)
      return 1.0;
  }
  // default value, used if there are no leptons associated to this jet
  return - std::numeric_limits<float>::infinity();
}
