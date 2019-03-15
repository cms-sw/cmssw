#include "RecoEgamma/ElectronIdentification/interface/ElectronMVAEstimatorRun2.h"

ElectronMVAEstimatorRun2::ElectronMVAEstimatorRun2(const edm::ParameterSet& conf):
  AnyMVAEstimatorRun2Base(conf),
  name_(conf.getParameter<std::string>("mvaName")),
  tag_(conf.getParameter<std::string>("mvaTag")),
  nCategories_            (conf.getParameter<int>("nCategories")),
  methodName_             ("BDTG method"),
  mvaVarMngr_(conf.getParameter<std::string>("variableDefinition")),
  debug_(conf.getUntrackedParameter<bool>("debug", false))
{

  const std::vector <std::string> weightFileNames
    = conf.getParameter<std::vector<std::string> >("weightFileNames");

  const std::vector <std::string> categoryCutStrings
    = conf.getParameter<std::vector<std::string> >("categoryCuts");

  if( (int)(categoryCutStrings.size()) != nCategories_ )
    throw cms::Exception("MVA config failure: ")
      << "wrong number of category cuts in " << name_ << tag_ << std::endl;

  for (int i = 0; i < nCategories_; ++i) {
      StringCutObjectSelector<reco::GsfElectron> select(categoryCutStrings[i]);
      categoryFunctions_.push_back(select);
  }

  // Initialize GBRForests from weight files
  init(weightFileNames);

}

ElectronMVAEstimatorRun2::ElectronMVAEstimatorRun2(
        const std::string &mvaTag, const std::string &mvaName, const bool debug):
  AnyMVAEstimatorRun2Base( edm::ParameterSet() ),
  name_                   (mvaName),
  tag_                    (mvaTag),
  methodName_             ("BDTG method"),
  debug_                  (debug) {
  }

void ElectronMVAEstimatorRun2::init(const std::vector<std::string> &weightFileNames) {

  if(debug_) {
    std::cout << " *** Inside " << name_ << tag_ << std::endl;
  }

  // Initialize GBRForests
  if( (int)(weightFileNames.size()) != nCategories_ )
    throw cms::Exception("MVA config failure: ")
      << "wrong number of weightfiles in " << name_ << tag_ << std::endl;

  gbrForests_.clear();
  // Create a TMVA reader object for each category
  for(int i=0; i<nCategories_; i++){

    std::vector<int> variablesInCategory;

    // Use unique_ptr so that all readers are properly cleaned up
    // when the vector clear() is called in the destructor

    std::vector<std::string> variableNamesInCategory;
    gbrForests_.push_back(createGBRForest(weightFileNames[i], variableNamesInCategory));

    nVariables_.push_back(variableNamesInCategory.size());

    variables_.push_back(variablesInCategory);

    if(debug_) {
      std::cout << " *** Inside " << name_ << tag_ << std::endl;
      std::cout << " category " << i << " with nVariables " << nVariables_[i] << std::endl;
    }

    for (int j=0; j<nVariables_[i];++j) {
        int index = mvaVarMngr_.getVarIndex(variableNamesInCategory[j]);
        if(index == -1) {
            throw cms::Exception("MVA config failure: ")
               << "Concerning " << name_ << tag_ << std::endl
               << "Variable " << variableNamesInCategory[j]
               << " not found in variable definition file!" << std::endl;
        }
        variables_[i].push_back(index);

    }
  }
}

ElectronMVAEstimatorRun2::
~ElectronMVAEstimatorRun2(){
}

void ElectronMVAEstimatorRun2::setConsumes(edm::ConsumesCollector&& cc) const {
  // All tokens for event content needed by this MVA
  // Tags from the variable helper
  for (auto &tag : mvaVarMngr_.getHelperInputTags()) {
      cc.consumes<edm::ValueMap<float>>(tag);
  }
  for (auto &tag : mvaVarMngr_.getGlobalInputTags()) {
      cc.consumes<double>(tag);
  }
}

float ElectronMVAEstimatorRun2::
mvaValue( const edm::Ptr<reco::Candidate>& candPtr, const edm::EventBase & iEvent) const {

  const int iCategory = findCategory( candPtr );

  if (iCategory < 0) return -999;

  std::vector<float> vars;

  const edm::Ptr<reco::GsfElectron> gsfPtr{ candPtr };

  if( gsfPtr.get() == nullptr ) {
    throw cms::Exception("MVA failure: ")
      << " given particle is expected to be reco::GsfElectron or pat::Electron," << std::endl
      << " but appears to be neither" << std::endl;
  }

  for (int i = 0; i < nVariables_[iCategory]; ++i) {
      vars.push_back(mvaVarMngr_.getValue(variables_[iCategory][i], gsfPtr, iEvent));
  }

  if(debug_) {
    std::cout << " *** Inside " << name_ << tag_ << std::endl;
    std::cout << " category " << iCategory << std::endl;
    for (int i = 0; i < nVariables_[iCategory]; ++i) {
        std::cout << " " << mvaVarMngr_.getName(variables_[iCategory][i]) << " " << vars[i] << std::endl;
    }
  }
  const float response = gbrForests_.at(iCategory)->GetResponse(vars.data()); // The BDT score

  if(debug_) {
    std::cout << " ### MVA " << response << std::endl << std::endl;
  }

  return response;
}

int ElectronMVAEstimatorRun2::findCategory( const edm::Ptr<reco::Candidate>& candPtr) const {

  auto gsfEle = dynamic_cast<reco::GsfElectron const*>(candPtr.get());

  if( gsfEle == nullptr ) {
    throw cms::Exception("MVA failure: ")
      << " given particle is expected to be reco::GsfElectron or pat::Electron," << std::endl
      << " but appears to be neither" << std::endl;
  }

  for (int i = 0; i < nCategories_; ++i) {
      if (categoryFunctions_[i](*gsfEle)) return i;
  }

  edm::LogWarning  ("MVA warning") <<
      "category not defined for particle with pt " << gsfEle->pt() << " GeV, eta " <<
          gsfEle->superCluster()->eta() << " in " << name_ << tag_;

  return -1;

}
