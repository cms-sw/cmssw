#include <limits>

#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "RecoBTag/SoftLepton/interface/LeptonSelector.h"
#include "RecoBTag/SoftLepton/interface/MuonTagger.h"

#include "TRandom3.h"

/// b-tag a jet based on track-to-jet parameters in the extened info collection
float MuonTagger::discriminator(const TagInfoHelper & tagInfo) const {
  MuonTaggerMLP theNet{};
  // default value, used if there are no leptons associated to this jet
  float bestTag = - std::numeric_limits<float>::infinity();
  const reco::SoftLeptonTagInfo & info = tagInfo.get<reco::SoftLeptonTagInfo>();
  // if there are multiple leptons, look for the highest tag result
  for (unsigned int i = 0; i < info.leptons(); i++) {
    const reco::SoftLeptonProperties & properties = info.properties(i);
    if (m_selector(properties)) {
			int theSeed=1+round(10000.0*fabs(properties.deltaR));
                        TRandom3 *r = new TRandom3(theSeed);
			float rndm = r->Uniform(0,1);
			//for negative tagger, flip 50% of the negative signs to positive value
			float sip3d = (m_selector.isNegative() && rndm<0.5) ? -properties.sip3d : properties.sip3d;
			float tag = theNet.Value(0, properties.ptRel, sip3d, properties.deltaR, properties.ratioRel);
			if (tag > bestTag) bestTag = tag;
			delete r;
    }
  }
  return bestTag;
}
