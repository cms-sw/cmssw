#include <typeinfo>

#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "RecoBTag/SoftLepton/interface/LeptonTaggerDistance.h"

/// b-tag a jet based on track-to-jet pseudo-angular distance
float LeptonTaggerDistance::discriminator(const reco::BaseTagInfo & tagInfo) const {
  try {
    const reco::SoftLeptonTagInfo & info = dynamic_cast<const reco::SoftLeptonTagInfo &>(tagInfo);
    // if there are any leptons, look for any one within the requested deltaR
    for (unsigned int i = 0; i < info.leptons(); i++) {
      if (info.properties(i).deltaR <= m_maxDistance)
        return 1.0;
    }
  } catch(std::bad_cast e) {
    // ERROR - trying to use the wrong XxxTagInfo
    throw edm::Exception(edm::errors::LogicError) << "Wrong reco::BaseTagInfo-derived collection passed, expected reco::SoftLeptonTagInfo";
  }
  // default value, used if there are no leptons associated to this jet
  return -1.0;
}
