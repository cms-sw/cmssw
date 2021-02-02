#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CommonTools/MVAUtils/interface/GBRForestTools.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "RecoEgamma/EgammaTools/interface/AnyMVAEstimatorRun2Base.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "CommonTools/Egamma/interface/EffectiveAreas.h"
#include "CondFormats/GBRForest/interface/GBRForest.h"
#include "RecoEgamma/EgammaTools/interface/MVAVariableHelper.h"
#include "RecoEgamma/EgammaTools/interface/MVAVariableManager.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/ThreadSafeFunctor.h"

class PhotonMVAEstimator : public AnyMVAEstimatorRun2Base {
public:
  // Constructor and destructor
  PhotonMVAEstimator(const edm::ParameterSet& conf);
  ~PhotonMVAEstimator() override{};

  // Calculation of the MVA value
  float mvaValue(const reco::Candidate* candPtr, std::vector<float> const& auxVars, int& iCategory) const override;

  int findCategory(const reco::Candidate* candPtr) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  int findCategory(reco::Photon const& photon) const;

  // The number of categories and number of variables per category
  int nCategories_;
  std::vector<ThreadSafeFunctor<StringCutObjectSelector<reco::Photon>>> categoryFunctions_;
  std::vector<int> nVariables_;

  // Data members
  std::vector<std::unique_ptr<const GBRForest>> gbrForests_;

  // There might be different variables for each category, so the variables
  // names vector is itself a vector of length nCategories
  std::vector<std::vector<int>> variables_;

  // The variable manager which stores how to obtain the variables
  MVAVariableManager<reco::Photon> mvaVarMngr_;

  // Other objects needed by the MVA
  std::unique_ptr<EffectiveAreas> effectiveAreas_;
  std::vector<double> phoIsoPtScalingCoeff_;
  double phoIsoCutoff_;
};

PhotonMVAEstimator::PhotonMVAEstimator(const edm::ParameterSet& conf)
    : AnyMVAEstimatorRun2Base(conf),
      mvaVarMngr_(conf.getParameter<std::string>("variableDefinition"), MVAVariableHelper::indexMap()) {
  //
  // Construct the MVA estimators
  //
  if (getTag() == "Run2Spring16NonTrigV1") {
    effectiveAreas_ =
        std::make_unique<EffectiveAreas>((conf.getParameter<edm::FileInPath>("effAreasConfigFile")).fullPath());
    phoIsoPtScalingCoeff_ = conf.getParameter<std::vector<double>>("phoIsoPtScalingCoeff");
    phoIsoCutoff_ = conf.getParameter<double>("phoIsoCutoff");
  }

  const auto weightFileNames = conf.getParameter<std::vector<std::string>>("weightFileNames");
  const auto categoryCutStrings = conf.getParameter<std::vector<std::string>>("categoryCuts");

  if ((int)(categoryCutStrings.size()) != getNCategories())
    throw cms::Exception("MVA config failure: ")
        << "wrong number of category cuts in PhotonMVAEstimator" << getTag() << std::endl;

  for (auto const& cut : categoryCutStrings)
    categoryFunctions_.emplace_back(cut);

  // Initialize GBRForests
  if (static_cast<int>(weightFileNames.size()) != getNCategories())
    throw cms::Exception("MVA config failure: ")
        << "wrong number of weightfiles in PhotonMVAEstimator" << getTag() << std::endl;

  gbrForests_.clear();
  // Create a TMVA reader object for each category
  for (int i = 0; i < getNCategories(); i++) {
    std::vector<int> variablesInCategory;

    std::vector<std::string> variableNamesInCategory;
    gbrForests_.push_back(createGBRForest(weightFileNames[i], variableNamesInCategory));

    nVariables_.push_back(variableNamesInCategory.size());

    variables_.push_back(variablesInCategory);

    for (int j = 0; j < nVariables_[i]; ++j) {
      int index = mvaVarMngr_.getVarIndex(variableNamesInCategory[j]);
      if (index == -1) {
        throw cms::Exception("MVA config failure: ")
            << "Concerning PhotonMVAEstimator" << getTag() << std::endl
            << "Variable " << variableNamesInCategory[j] << " not found in variable definition file!" << std::endl;
      }
      variables_[i].push_back(index);
    }
  }
}

float PhotonMVAEstimator::mvaValue(const reco::Candidate* candPtr,
                                   std::vector<float> const& auxVars,
                                   int& iCategory) const {
  const reco::Photon* phoPtr = dynamic_cast<const reco::Photon*>(candPtr);
  if (phoPtr == nullptr) {
    throw cms::Exception("MVA failure: ")
        << " given particle is expected to be reco::Photon or pat::Photon," << std::endl
        << " but appears to be neither" << std::endl;
  }

  iCategory = findCategory(phoPtr);

  std::vector<float> vars;

  vars.reserve(nVariables_[iCategory]);
  for (int i = 0; i < nVariables_[iCategory]; ++i) {
    vars.push_back(mvaVarMngr_.getValue(variables_[iCategory][i], *phoPtr, auxVars));
  }

  // Special case for Spring16!
  if (getTag() == "Run2Spring16NonTrigV1" and iCategory == 1) {  // Endcap category
    // Raw value for EB only, because of loss of transparency in EE
    // for endcap MVA only in 2016
    double eA = effectiveAreas_->getEffectiveArea(std::abs(phoPtr->superCluster()->eta()));
    double phoIsoCorr = vars[10] - eA * (double)vars[9] - phoIsoPtScalingCoeff_.at(1) * phoPtr->pt();
    vars[10] = TMath::Max(phoIsoCorr, phoIsoCutoff_);
  }

  if (isDebug()) {
    std::cout << " *** Inside PhotonMVAEstimator" << getTag() << std::endl;
    std::cout << " category " << iCategory << std::endl;
    for (int i = 0; i < nVariables_[iCategory]; ++i) {
      std::cout << " " << mvaVarMngr_.getName(variables_[iCategory][i]) << " " << vars[i] << std::endl;
    }
  }

  const float response = gbrForests_.at(iCategory)->GetResponse(vars.data());

  if (isDebug()) {
    std::cout << " ### MVA " << response << std::endl << std::endl;
  }

  return response;
}

int PhotonMVAEstimator::findCategory(const reco::Candidate* candPtr) const {
  const reco::Photon* phoPtr = dynamic_cast<const reco::Photon*>(candPtr);
  if (phoPtr == nullptr) {
    throw cms::Exception("MVA failure: ")
        << " given particle is expected to be reco::Photon or pat::Photon," << std::endl
        << " but appears to be neither" << std::endl;
  }

  return findCategory(*phoPtr);
}

int PhotonMVAEstimator::findCategory(reco::Photon const& photon) const {
  for (int i = 0; i < getNCategories(); ++i) {
    if (categoryFunctions_[i](photon))
      return i;
  }

  edm::LogWarning("MVA warning") << "category not defined for particle with pt " << photon.pt() << " GeV, eta "
                                 << photon.superCluster()->eta() << " in PhotonMVAEstimator" << getTag();

  return -1;
}

DEFINE_EDM_PLUGIN(AnyMVAEstimatorRun2Factory, PhotonMVAEstimator, "PhotonMVAEstimator");
