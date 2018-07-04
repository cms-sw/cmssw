#ifndef RecoEgamma_ElectronIdentification_ElectronMVAEstimatorRun2_H
#define RecoEgamma_ElectronIdentification_ElectronMVAEstimatorRun2_H

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "RecoEgamma/EgammaTools/interface/AnyMVAEstimatorRun2Base.h"
#include "RecoEgamma/EgammaTools/interface/GBRForestTools.h"
#include "RecoEgamma/EgammaTools/interface/MVAVariableManager.h"

class ElectronMVAEstimatorRun2 : public AnyMVAEstimatorRun2Base{

 public:

  // Constructor and destructor
  ElectronMVAEstimatorRun2(const edm::ParameterSet& conf);
  ~ElectronMVAEstimatorRun2() override;
  // For use with FWLite/Python
  ElectronMVAEstimatorRun2(const std::string &mvaTag,
                           const std::string &mvaName,
                           const bool debug = false);

  void init(const std::vector<std::string> &weightFileNames);

  // Calculation of the MVA value
  float mvaValue( const edm::Ptr<reco::Candidate>& candPtr, const edm::EventBase& iEvent) const override;

  // Utility functions
  int getNCategories() const override { return nCategories_; }
  const std::string& getName() const final { return name_; }
  const std::string& getTag() const final { return tag_; }

  int findCategory( const edm::Ptr<reco::Candidate>& candPtr) const override;

  // Call this function once after the constructor to declare
  // the needed event content pieces to the framework
  void setConsumes(edm::ConsumesCollector&&) const final;

 private:

  // MVA name. This is a unique name for this MVA implementation.
  // It will be used as part of ValueMap names.
  // For simplicity, keep it set to the class name.
  const std::string name_;

  // MVA tag. This is an additional string variable to distinguish
  // instances of the estimator of this class configured with different
  // weight files.
  const std::string tag_;

  // The number of categories and number of variables per category
  int nCategories_;
  std::vector<StringCutObjectSelector<reco::GsfElectron>> categoryFunctions_;
  std::vector<int> nVariables_;

  // Data members
  std::vector< std::unique_ptr<const GBRForest> > gbrForests_;

  const std::string methodName_;


  // There might be different variables for each category, so the variables
  // names vector is itself a vector of length nCategories
  std::vector<std::vector<int>> variables_;

  MVAVariableManager<reco::GsfElectron> mvaVarMngr_;

  bool debug_;

};

#endif
