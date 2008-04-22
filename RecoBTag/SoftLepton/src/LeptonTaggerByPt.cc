#include <typeinfo>

#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "RecoBTag/SoftLepton/interface/LeptonTaggerByPt.h"

/// b-tag a jet based on track-to-jet parameters in the extened info collection
float LeptonTaggerByPt::discriminator(const TagInfoHelper & tagInfo) const {
  // default value, used if there are no leptons associated to this jet
  float bestTag = 0.;
  const reco::SoftLeptonTagInfo & info = tagInfo.get<reco::SoftLeptonTagInfo>();
  // if there are multiple leptons, look for the one with the highest pT_rel
  for (unsigned int i = 0; i < info.leptons(); i++) {
    float tag = info.properties(i).ptRel;
    if (tag > bestTag)
      bestTag = tag;
  }
  return bestTag;
}
