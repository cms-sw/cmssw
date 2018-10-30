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
  ~ElectronMVAEstimatorRun2() override {};
  // For use with FWLite/Python
  ElectronMVAEstimatorRun2(const std::string &mvaTag,
                           const std::string &mvaName,
                           const bool debug = false);

  void init(const std::vector<std::string> &weightFileNames);

  // Calculation of the MVA value
  float mvaValue( const edm::Ptr<reco::Candidate>& candPtr, const edm::EventBase& iEvent, int &iCategory) const override;

  // Call this function once after the constructor to declare
  // the needed event content pieces to the framework
  void setConsumes(edm::ConsumesCollector&&) final;

  int findCategory( const edm::Ptr<reco::Candidate>& candPtr) const override;

 private:

  int findCategory( const edm::Ptr<reco::GsfElectron>& gsfPtr) const;

  std::vector<StringCutObjectSelector<reco::GsfElectron>> categoryFunctions_;
  std::vector<int> nVariables_;

  // Data members
  std::vector< std::unique_ptr<const GBRForest> > gbrForests_;


  // There might be different variables for each category, so the variables
  // names vector is itself a vector of length nCategories
  std::vector<std::vector<int>> variables_;

  MVAVariableManager<reco::GsfElectron> mvaVarMngr_;

};

#endif
