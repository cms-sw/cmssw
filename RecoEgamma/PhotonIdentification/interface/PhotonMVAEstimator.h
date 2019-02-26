#ifndef RecoEgamma_PhotonIdentification_PhotonMVAEstimator_H
#define RecoEgamma_PhotonIdentification_PhotonMVAEstimator_H

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "RecoEgamma/EgammaTools/interface/AnyMVAEstimatorRun2Base.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "RecoEgamma/EgammaTools/interface/EffectiveAreas.h"
#include "CondFormats/EgammaObjects/interface/GBRForest.h"
#include "RecoEgamma/EgammaTools/interface/MVAVariableManager.h"
#include "RecoEgamma/EgammaTools/interface/ThreadSafeStringCut.h"

class PhotonMVAEstimator : public AnyMVAEstimatorRun2Base{
  
 public:

  // Constructor and destructor
  PhotonMVAEstimator(const edm::ParameterSet& conf);
  ~PhotonMVAEstimator() override {};

  // Calculation of the MVA value
  float mvaValue( const reco::Candidate* candPtr, std::vector<float> const& auxVars, int &iCategory) const override;
  
  int findCategory( const reco::Candidate* candPtr) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:

  int findCategory(reco::Photon const& photon) const;

  // The number of categories and number of variables per category
  int nCategories_;
  std::vector<ThreadSafeStringCut<StringCutObjectSelector<reco::Photon>,reco::Photon>> categoryFunctions_;
  std::vector<int> nVariables_;

  // Data members
  std::vector< std::unique_ptr<const GBRForest> > gbrForests_;

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

#endif
