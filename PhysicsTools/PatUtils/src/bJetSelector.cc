//
//  Implementacion of b Jet Selector for PAT 
//   By J.E. Ramirez Jun 18,2008
//
#include "PhysicsTools/PatUtils/interface/bJetSelector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

bJetSelector::bJetSelector(const edm::ParameterSet& cfg) :
  discriminantCutsLoose_(cfg.getParameter<std::vector<double> >("discCutLoose")),
  discriminantCutsMedium_(cfg.getParameter<std::vector<double> >("discCutMedium")),
  discriminantCutsTight_(cfg.getParameter<std::vector<double> >("discCutTight")),
  BTagdiscriminator_(cfg.getParameter<std::vector<std::string> >("bdiscriminators")), 
  DefaultOp_(cfg.getParameter<std::string>("DefaultOp")),
  DefaultTg_(cfg.getParameter<std::string>("DefaultBdisc"))

{

  for (unsigned int i=0; i<BTagdiscriminator_.size(); i++){
     discCut["Loose"][BTagdiscriminator_[i]] = discriminantCutsLoose_[i];
     discCut["Medium"][BTagdiscriminator_[i]] = discriminantCutsMedium_[i];
     discCut["Tight"][BTagdiscriminator_[i]] = discriminantCutsTight_[i];
  } 
}

bool
bJetSelector::IsbTag(const pat::Jet& JetCand, 
                          const std::string& operpoint, 
                          const std::string& tagger) const {

	std::map<std::string,std::map<std::string,double> >::const_iterator ioperpoint = discCut.find(operpoint);
	if ( ioperpoint == discCut.end() ) throw cms::Exception("UnknownOperatingPoint") << "Unknown or undefined operative point" << std::endl;
	std::map<std::string,double>::const_iterator itagger = ioperpoint->second.find(tagger);
	if ( itagger == ioperpoint->second.end() ) throw cms::Exception("UnknownTagger") << "Unknown or undefined tagger" << std::endl;
	
	return JetCand.bDiscriminator(tagger) > itagger->second;
}
bool
bJetSelector::IsbTag(const pat::Jet& JetCand, 
					 const std::string& operpoint) const {

	
        return IsbTag(JetCand,operpoint,DefaultTg_);
}
bool
bJetSelector::IsbTag(const pat::Jet& JetCand) const{ 
        return IsbTag(JetCand,DefaultOp_,DefaultTg_);
}


