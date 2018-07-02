#ifndef RecoEgamma_PhotonIdentification_PhotonMVAEstimator_H
#define RecoEgamma_PhotonIdentification_PhotonMVAEstimator_H

#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/Common/interface/ValueMap.h"

#include "RecoEgamma/EgammaTools/interface/AnyMVAEstimatorRun2Base.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "RecoEgamma/EgammaTools/interface/GBRForestTools.h"
#include "RecoEgamma/EgammaTools/interface/MVAVariableManager.h"

#include "RecoEgamma/EgammaTools/interface/EffectiveAreas.h"

#include <vector>
#include <string>

class PhotonMVAEstimator : public AnyMVAEstimatorRun2Base{
  
 public:

  // Define here the number and the meaning of the categories
  // for this specific MVA
  const int nCategories_ = 2;
  enum mvaCategories {
    CAT_EB  = 0,
    CAT_EE  = 1
  };

  // Constructor and destructor
  PhotonMVAEstimator(const edm::ParameterSet& conf);
  ~PhotonMVAEstimator() override;

  // Calculation of the MVA value
  float mvaValue( const edm::Ptr<reco::Candidate>& candPtr, const edm::EventBase&) const override;
 
  // Utility functions
  std::unique_ptr<const GBRForest> createSingleReader(const int iCategory, const edm::FileInPath &weightFile);
  
  int getNCategories() const override { return nCategories_; }
  const std::string& getName() const final { return name_; }
  const std::string& getTag() const final { return tag_; }
  
  int findCategory( const edm::Ptr<reco::Candidate>& candPtr ) const override;

  // Call this function once after the constructor to declare
  // the needed event content pieces to the framework
  void setConsumes(edm::ConsumesCollector&&) const override;
  
 private:

  // MVA name. This is a unique name for this MVA implementation.
  // It will be used as part of ValueMap names.
  // For simplicity, keep it set to the class name.
  const std::string name_;

  // MVA tag. This is an additional string variable to distinguish
  // instances of the estimator of this class configured with different
  // weight files.
  const std::string tag_;

  std::vector<int> nVariables_;

  // Data members
  std::vector< std::unique_ptr<const GBRForest> > gbrForests_;

  // All variables needed by this MVA
  const std::string methodName_;

  // There might be different variables for each category, so the variables
  // names vector is itself a vector of length nCategories
  std::vector<std::vector<int>> variables_;

  // The variable manager which stores how to obtain the variables
  MVAVariableManager<reco::Photon> mvaVarMngr_;

  const double ebeeSplit_;

  const bool debug_;

  // Other objects needed by the MVA
  std::unique_ptr<EffectiveAreas> effectiveAreas_;
  std::vector<double> phoIsoPtScalingCoeff_;
  double phoIsoCutoff_;

};

#endif
