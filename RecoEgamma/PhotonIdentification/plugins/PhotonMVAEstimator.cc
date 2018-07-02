#include "RecoEgamma/PhotonIdentification/interface/PhotonMVAEstimator.h"

PhotonMVAEstimator::PhotonMVAEstimator(const edm::ParameterSet& conf):
  AnyMVAEstimatorRun2Base(conf),
  name_(conf.getParameter<std::string>("mvaName")),
  tag_(conf.getParameter<std::string>("mvaTag")),
  methodName_("BDTG method"),
  mvaVarMngr_(conf.getParameter<std::string>("variableDefinition")),
  ebeeSplit_ (conf.getParameter<double> ("ebeeSplit")),
  debug_(conf.getUntrackedParameter<bool>("debug", false))
{
  //
  // Construct the MVA estimators
  //
  if (tag_ == "Run2Spring16NonTrigV1") {
      effectiveAreas_ = std::make_unique<EffectiveAreas>((conf.getParameter<edm::FileInPath>("effAreasConfigFile")).fullPath());
      phoIsoPtScalingCoeff_ = conf.getParameter<std::vector<double >>("phoIsoPtScalingCoeff");
      phoIsoCutoff_ = conf.getParameter<double>("phoIsoCutoff");
  }

  const std::vector <std::string> weightFileNames
    = conf.getParameter<std::vector<std::string> >("weightFileNames");

  // Initialize GBRForests
  if( (int)(weightFileNames.size()) != nCategories_ )
    throw cms::Exception("MVA config failure: ")
      << "wrong number of weightfiles in " << name_ << tag_ << std::endl;

  gbrForests_.clear();
  // Create a TMVA reader object for each category
  for(int i=0; i<nCategories_; i++){

    std::vector<std::string> variableNamesInCategory;
    std::vector<int> variablesInCategory;

    // Use unique_ptr so that all readers are properly cleaned up
    // when the vector clear() is called in the destructor

    gbrForests_.push_back( GBRForestTools::createGBRForest( weightFileNames[i], variableNamesInCategory ) );

    nVariables_.push_back(variableNamesInCategory.size());

    variables_.push_back(variablesInCategory);

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

PhotonMVAEstimator::
~PhotonMVAEstimator(){
}

float PhotonMVAEstimator::
mvaValue(const edm::Ptr<reco::Candidate>& candPtr, const edm::EventBase& iEvent) const {

  const int iCategory = findCategory( candPtr );

  const edm::Ptr<reco::Photon> phoPtr{ candPtr };
  if( phoPtr.get() == nullptr) {
    throw cms::Exception("MVA failure: ")
      << " given particle is expected to be reco::Photon or pat::Photon," << std::endl
      << " but appears to be neither" << std::endl;
  }

  std::vector<float> vars;

  for (int i = 0; i < nVariables_[iCategory]; ++i) {
      vars.push_back(mvaVarMngr_.getValue(variables_[iCategory][i], phoPtr, iEvent));
  }

  // Special case for Spring16!
  if (tag_ == "Run2Spring16NonTrigV1" and iCategory == CAT_EE) {
      // Raw value for EB only, because of loss of transparency in EE
      // for endcap MVA only in 2016
      double eA = effectiveAreas_->getEffectiveArea( std::abs(phoPtr->superCluster()->eta()) );
      double phoIsoCorr = vars[10] - eA*(double)vars[9] - phoIsoPtScalingCoeff_.at(1) * phoPtr->pt();
      vars[10] = TMath::Max( phoIsoCorr, phoIsoCutoff_);
  }

  if(debug_) {
    std::cout << " *** Inside " << name_ << tag_ << std::endl;
    std::cout << " category " << iCategory << std::endl;
    for (int i = 0; i < nVariables_[iCategory]; ++i) {
        std::cout << " " << mvaVarMngr_.getName(variables_[iCategory][i]) << " " << vars[i] << std::endl;
    }
  }

  const float response = gbrForests_.at(iCategory)->GetResponse(vars.data());

  if(debug_) {
    std::cout << " ### MVA " << response << std::endl << std::endl;
  }

  return response;
}

int PhotonMVAEstimator::findCategory( const edm::Ptr<reco::Candidate>& candPtr) const {

  // Try to cast the particle into a reco particle.
  // This should work for both reco and pat.
  auto pho = dynamic_cast<reco::Photon const*>(candPtr.get());

  if( pho == nullptr) {
    throw cms::Exception("MVA failure: ")
      << " given particle is expected to be reco::Photon or pat::Photon," << std::endl
      << " but appears to be neither" << std::endl;
  }

  float eta = pho->superCluster()->eta();

  //
  // Determine the category
  //
  if ( std::abs(eta) < ebeeSplit_)
     return CAT_EB; 
  else
     return CAT_EE;
}

void PhotonMVAEstimator::setConsumes(edm::ConsumesCollector&& cc) const {
  // All tokens for event content needed by this MVA
  // Tags from the variable helper
  for (auto &tag : mvaVarMngr_.getHelperInputTags()) {
      cc.consumes<edm::ValueMap<float>>(tag);
  }
  for (auto &tag : mvaVarMngr_.getGlobalInputTags()) {
      cc.consumes<double>(tag);
  }
}

DEFINE_EDM_PLUGIN(AnyMVAEstimatorRun2Factory,
          PhotonMVAEstimator,
          "PhotonMVAEstimator");
