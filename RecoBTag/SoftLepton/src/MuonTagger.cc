// * Author: Alberto Zucchetta
// * Mail: a.zucchetta@cern.ch
// * January 16, 2015

#include <limits>
#include <random>

#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "DataFormats/BTauReco/interface/CandSoftLeptonTagInfo.h"
#include "RecoBTag/SoftLepton/interface/LeptonSelector.h"
#include "RecoBTag/SoftLepton/interface/MuonTagger.h"


MuonTagger::MuonTagger(const edm::ParameterSet& conf): m_selector(conf) {
  uses("smTagInfos");
  edm::FileInPath WeightFile=conf.getParameter<edm::FileInPath>("weightFile");
  mvaID.reset(new TMVAEvaluator());

  // variable order needs to be the same as in the training
  std::vector<std::string> variables({"TagInfo1.sip3d", "TagInfo1.sip2d", "TagInfo1.ptRel", "TagInfo1.deltaR", "TagInfo1.ratio"});
  std::vector<std::string> spectators;

  mvaID->initialize("Color:Silent:Error", "BDT", WeightFile.fullPath(), variables, spectators);
}


// b-tag a jet based on track-to-jet parameters in the extened info collection
float MuonTagger::discriminator(const TagInfoHelper& tagInfo) const {

  float bestTag = - std::numeric_limits<float>::infinity(); // default value, used if there are no leptons associated to this jet
  const reco::CandSoftLeptonTagInfo& info = tagInfo.get<reco::CandSoftLeptonTagInfo>();

  std::mt19937_64 random;
  std::uniform_real_distribution<float> dist(0.f,1.f);

  // TMVAEvaluator is not thread safe
  std::lock_guard<std::mutex> lock(m_mutex);
  
  // If there are multiple leptons, look for the highest tag result
  for (unsigned int i=0; i<info.leptons(); i++) {
    const reco::SoftLeptonProperties& properties = info.properties(i);
    bool flip(false);
    if(m_selector.isNegative()) {
      int seed=1+round(10000.*properties.deltaR);
      random.seed(seed);
      float rndm = dist(random);
      if(rndm<0.5) flip=true;
    }
    //for negative tagger, flip 50% of the negative signs to positive value
    float sip3d = flip ? -properties.sip3d : properties.sip3d;
    float sip2d = flip ? -properties.sip2d : properties.sip2d;

    std::map<std::string,float> inputs;
    inputs["TagInfo1.sip3d"] = sip3d;
    inputs["TagInfo1.sip2d"] = sip2d;
    inputs["TagInfo1.ptRel"] = properties.ptRel;
    inputs["TagInfo1.deltaR"] = properties.deltaR;
    inputs["TagInfo1.ratio"] = properties.ratio;

    float tag = mvaID->evaluate(inputs);
    // Transform output between 0 and 1
    tag = (tag+1.0)/2.0;
    if(tag>bestTag) bestTag = tag;
  }

  return bestTag;
}

