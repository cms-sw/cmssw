#include <limits>

#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "RecoBTag/SoftLepton/interface/LeptonSelector.h"
#include "RecoBTag/SoftLepton/interface/MuonTagger.h"

/// b-tag a jet based on track-to-jet parameters in the extened info collection
float MuonTagger::discriminator(const TagInfoHelper & tagInfo) const {
  // default value, used if there are no leptons associated to this jet
  float bestTag = - std::numeric_limits<float>::infinity();
  const reco::SoftLeptonTagInfo & info = tagInfo.get<reco::SoftLeptonTagInfo>();
  // if there are multiple leptons, look for the highest tag result
  for (unsigned int i = 0; i < info.leptons(); i++) {
    const reco::SoftLeptonProperties & properties = info.properties(i);
    if (m_selector(properties)) {
      float sip3d = m_selector.isNegative() ? -properties.sip3d : properties.sip3d;
      float tag = theNet.value( 0, properties.ptRel, properties.ratioRel, properties.deltaR, info.jet()->energy(), info.jet()->eta(), sip3d) +
                  theNet.value( 1, properties.ptRel, properties.ratioRel, properties.deltaR, info.jet()->energy(), info.jet()->eta(), sip3d);
      if (tag > bestTag)
        bestTag = tag;
    }
  }
  return bestTag;
}
