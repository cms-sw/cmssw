#include <limits>
#include <random>

#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "RecoBTag/SoftLepton/interface/LeptonSelector.h"
#include "RecoBTag/SoftLepton/interface/ElectronTagger.h"
#include "DataFormats/BTauReco/interface/CandSoftLeptonTagInfo.h"
#include <iostream>

ElectronTagger::ElectronTagger(const edm::ParameterSet & configuration):
    m_selector(configuration)
  {
	uses("seTagInfos");
	edm::FileInPath WeightFile=configuration.getParameter<edm::FileInPath>("weightFile");
	mvaID.reset(new TMVAEvaluator());
	
	// variable order needs to be the same as in the training
	std::vector<std::string> variables({"sip3d", "sip2d", "ptRel", "deltaR", "ratio", "mva_e_pi"});
	std::vector<std::string> spectators;
	
	mvaID->initialize("Color:Silent:Error", "BDT", WeightFile.fullPath(), variables, spectators);
  }

/// b-tag a jet based on track-to-jet parameters in the extened info collection
float ElectronTagger::discriminator(const TagInfoHelper & tagInfo) const {
  // default value, used if there are no leptons associated to this jet
  float bestTag = - std::numeric_limits<float>::infinity();
  const reco::CandSoftLeptonTagInfo & info = tagInfo.get<reco::CandSoftLeptonTagInfo>();

  std::mt19937_64 random;
  std::uniform_real_distribution<float> dist(0.f,1.f);

  // TMVAEvaluator is not thread safe
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
	
	std::map<std::string,float> inputs;
	inputs["sip3d"] = sip3d;
	inputs["sip2d"] = properties.sip2d;
	inputs["ptRel"] = properties.ptRel;
	inputs["deltaR"] = properties.deltaR;
	inputs["ratio"] = properties.ratio;
	inputs["mva_e_pi"] = properties.elec_mva;
	
	float tag = mvaID->evaluate(inputs);
	// Transform output between 0 and 1
	tag = (tag+1.0)/2.0;
        if (tag > bestTag)
           bestTag = tag;
    }
  }
  return bestTag;
}
