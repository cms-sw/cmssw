// * Author: Alberto Zucchetta
// * Mail: a.zucchetta@cern.ch
// * January 16, 2015

#include <limits>

#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "DataFormats/BTauReco/interface/CandSoftLeptonTagInfo.h"
#include "RecoBTag/SoftLepton/interface/LeptonSelector.h"
#include "RecoBTag/SoftLepton/interface/MuonTagger.h"


MuonTagger::MuonTagger(const edm::ParameterSet& conf): m_selector(conf) {
  uses("smTagInfos");
  random=new TRandom3();
  mvaID=new MvaSoftMuonEstimator();
}

MuonTagger::~MuonTagger() {
  delete mvaID;
  delete random;
}


// b-tag a jet based on track-to-jet parameters in the extened info collection
float MuonTagger::discriminator(const TagInfoHelper& tagInfo) const {

  float bestTag = - std::numeric_limits<float>::infinity(); // default value, used if there are no leptons associated to this jet
  const reco::CandSoftLeptonTagInfo& info = tagInfo.get<reco::CandSoftLeptonTagInfo>();

  // If there are multiple leptons, look for the highest tag result
  for (unsigned int i=0; i<info.leptons(); i++) {
    const reco::SoftLeptonProperties& properties = info.properties(i);
    bool flip(false);
    if(m_selector.isNegative()) {
      int seed=1+round(10000.*fabs(properties.deltaR));
      random->SetSeed(seed);
      float rndm = random->Uniform(0,1);
      if(rndm<0.5) flip=true;
    }
    float sip3d = flip ? -properties.sip3d : properties.sip3d;
    float sip2d = flip ? -properties.sip2d : properties.sip2d;
    float tag = mvaID->mvaValue(sip3d, sip2d, properties.ptRel, properties.ratio);
    if(tag>bestTag) bestTag = tag;
  }
  
  return bestTag;
}

