#include <limits>

#include "DataFormats/BTauReco/interface/CandSoftLeptonTagInfo.h"
#include "RecoBTag/SoftLepton/interface/LeptonTaggerByIP.h"
#include "RecoBTag/SoftLepton/interface/LeptonSelector.h"

/// b-tag a jet based on track-to-jet parameters in the extened info collection
float LeptonTaggerByIP::discriminator(const TagInfoHelper & tagInfo) const {
  // default value, used if there are no leptons associated to this jet
  float bestTag = - std::numeric_limits<float>::infinity();
  const reco::CandSoftLeptonTagInfo & info = tagInfo.get<reco::CandSoftLeptonTagInfo>();
  // if there are multiple leptons, look for the one with the highest pT_rel
  for (unsigned int i = 0; i < info.leptons(); i++) {
    const reco::SoftLeptonProperties & properties = info.properties(i);
    float sipsig = m_use3d ? properties.sip3dsig : properties.sip2dsig;
    if (m_selector.isNegative())
      sipsig = -sipsig;
    if (m_selector(properties, m_use3d)) {
      float tag = sipsig;
      if (tag > bestTag)
        bestTag = tag;
    }
  }
  return bestTag;
}
