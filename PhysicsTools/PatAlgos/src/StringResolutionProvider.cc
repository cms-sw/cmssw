#include <map>
#include <Math/Functions.h>

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "PhysicsTools/PatAlgos/interface/StringResolutionProvider.h"
#include "DataFormats/PatCandidates/interface/ParametrizationHelper.h"

StringResolutionProvider::StringResolutionProvider(const edm::ParameterSet& cfg): constraints_() 
{ 
  typedef pat::CandKinResolution::Parametrization Parametrization;
  
  // 
  std::vector<double> constr = cfg.getParameter<std::vector<double> > ("constraints");
  constraints_.insert(constraints_.end(), constr.begin(), constr.end());
  
  std::string parametrization(cfg.getParameter<std::string> ("parametrization") );
  parametrization_ = pat::helper::ParametrizationHelper::fromString(parametrization);
  
  std::vector<edm::ParameterSet> functionSets_ = cfg.getParameter <std::vector<edm::ParameterSet> >("functions");
  for(std::vector<edm::ParameterSet>::const_iterator iSet = functionSets_.begin(); iSet != functionSets_.end(); ++iSet){
    if(iSet->exists("bin"))bins_.push_back(iSet->getParameter<std::string>("bin"));
    else if(functionSets_.size()==1)bins_.push_back("");
    else throw cms::Exception("WrongConfig") << "Parameter 'bin' is needed if more than one PSet is specified\n";

    funcEt_.push_back(iSet->getParameter<std::string>("et"));
    funcEta_.push_back(iSet->getParameter<std::string>("eta"));
    funcPhi_.push_back(iSet->getParameter<std::string>("phi"));
  }
}

StringResolutionProvider::~StringResolutionProvider()
{
}
 
pat::CandKinResolution
StringResolutionProvider::getResolution(const reco::Candidate& cand) const 
{ 
  int selectedBin=-1;
  for(unsigned int i=0; i<bins_.size(); ++i){
    StringCutObjectSelector<reco::Candidate> select_(bins_[i]);
    if(select_(cand)){
      selectedBin = i;
      break;
    }
  }
  std::vector<pat::CandKinResolution::Scalar> covariances(3);
  if(selectedBin>=0){
    covariances[0] = ROOT::Math::Square(Function(funcEt_ [selectedBin]).operator()(cand));
    covariances[1] = ROOT::Math::Square(Function(funcEta_[selectedBin]).operator()(cand));
    covariances[2] = ROOT::Math::Square(Function(funcPhi_[selectedBin]).operator()(cand));
  }
  // fill 0. for not selected candidates
  else for(int i=0; i<3; ++i){covariances[i] = 0.;}
  
  return pat::CandKinResolution(parametrization_, covariances, constraints_);
}
