#include <limits>
#include <random>

#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "RecoBTag/SoftLepton/interface/LeptonSelector.h"
#include "RecoBTag/SoftLepton/interface/ElectronTagger.h"
#include "DataFormats/BTauReco/interface/CandSoftLeptonTagInfo.h"
#include <iostream>

/// b-tag a jet based on track-to-jet parameters in the extened info collection
float ElectronTagger::discriminator(const TagInfoHelper & tagInfo) const {
  // default value, used if there are no leptons associated to this jet
  float bestTag = - std::numeric_limits<float>::infinity();
  const reco::CandSoftLeptonTagInfo & info = tagInfo.get<reco::CandSoftLeptonTagInfo>();

  std::mt19937_64 random;
  std::uniform_real_distribution<float> dist(0.f,1.f);

  //MvaSofEleEstimator is not thread safe
  std::lock_guard<std::mutex> lock(m_mutex);
  // if there are multiple leptons, look for the highest tag result
  for (unsigned int i = 0; i < info.leptons(); i++) {
    const reco::SoftLeptonProperties & properties = info.properties(i);
    if (m_selector(properties)) {
	int theSeed=1+round(10000.0*std::abs(properties.deltaR));
	random.seed(theSeed);
	float rndm = dist(random);
	//for negative tagger, flip 50% of the negative signs to positive value
	float sip3d = (m_selector.isNegative() && rndm<0.5) ? -properties.sip3d : properties.sip3d;
	float tag = mvaID->mvaValue( properties.sip2d, sip3d, properties.ptRel, properties.deltaR, properties.ratio,properties.elec_mva);
        if (tag > bestTag)
           bestTag = tag;
    }
  }
  return bestTag;
}
