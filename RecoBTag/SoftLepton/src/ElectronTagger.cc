#include <limits>
#include <random>

#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/DataRecord/interface/BTauGenericMVAJetTagComputerRcd.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputerRecord.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "RecoBTag/SoftLepton/interface/LeptonSelector.h"
#include "RecoBTag/SoftLepton/interface/ElectronTagger.h"
#include "DataFormats/BTauReco/interface/CandSoftLeptonTagInfo.h"
#include <iostream>

ElectronTagger::ElectronTagger(const edm::ParameterSet & cfg):
    m_selector(cfg),
    m_useCondDB(cfg.getParameter<bool>("useCondDB")),
    m_gbrForestLabel(cfg.existsAs<std::string>("gbrForestLabel") ? cfg.getParameter<std::string>("gbrForestLabel") : ""),
    m_weightFile(cfg.existsAs<edm::FileInPath>("weightFile") ? cfg.getParameter<edm::FileInPath>("weightFile") : edm::FileInPath()),
    m_useGBRForest(cfg.existsAs<bool>("useGBRForest") ? cfg.getParameter<bool>("useGBRForest") : false),
    m_useAdaBoost(cfg.existsAs<bool>("useAdaBoost") ? cfg.getParameter<bool>("useAdaBoost") : false)
  {
	uses("seTagInfos");
	mvaID.reset(new TMVAEvaluator());
  }

void ElectronTagger::initialize(const JetTagComputerRecord & record)
{
	// variable names and order need to be the same as in the training
	std::vector<std::string> variables({"sip3d", "sip2d", "ptRel", "deltaR", "ratio", "mva_e_pi"});
	std::vector<std::string> spectators;
	
	if (m_useCondDB)
	{
		const GBRWrapperRcd & gbrWrapperRecord = record.getRecord<GBRWrapperRcd>();
		
		edm::ESHandle<GBRForest> gbrForestHandle;
		gbrWrapperRecord.get(m_gbrForestLabel.c_str(), gbrForestHandle);

		mvaID->initializeGBRForest(gbrForestHandle.product(), variables, spectators, m_useAdaBoost);
	}
	else
		mvaID->initialize("Color:Silent:Error", "BDT", m_weightFile.fullPath(), variables, spectators, m_useGBRForest, m_useAdaBoost);
}

/// b-tag a jet based on track-to-jet parameters in the extened info collection
float ElectronTagger::discriminator(const TagInfoHelper & tagInfo) const {
  // default value, used if there are no leptons associated to this jet
  float bestTag = - std::numeric_limits<float>::infinity();
  const reco::CandSoftLeptonTagInfo & info = tagInfo.get<reco::CandSoftLeptonTagInfo>();

  std::mt19937_64 random;
  std::uniform_real_distribution<float> dist(0.f,1.f);

  // if there are multiple leptons, look for the highest tag result
  for (unsigned int i = 0; i < info.leptons(); i++) {
    const reco::SoftLeptonProperties & properties = info.properties(i);
    if (m_selector(properties)) {
	int theSeed=1+round(10000.0*std::abs(properties.deltaR));
	random.seed(theSeed);
	float rndm = dist(random);
	//for negative tagger, flip 50% of the negative signs to positive value
	float sip3dsig = (m_selector.isNegative() && rndm<0.5) ? -properties.sip3dsig : properties.sip3dsig;
	float sip2dsig = (m_selector.isNegative() && rndm<0.5) ? -properties.sip2dsig : properties.sip2dsig;
	
	std::map<std::string,float> inputs;
	inputs["sip3d"] = sip3dsig;
	inputs["sip2d"] = sip2dsig;
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
