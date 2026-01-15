#ifndef RecoEgamma_ElectronIdentification_ElectronMVAEstimatorRun2_H
#define RecoEgamma_ElectronIdentification_ElectronMVAEstimatorRun2_H

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/ThreadSafeFunctor.h"
#include "CondFormats/GBRForest/interface/GBRForest.h"
#include "RecoEgamma/EgammaTools/interface/AnyMVAEstimatorRun2Base.h"
#include "RecoEgamma/EgammaTools/interface/MVAVariableManager.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

// Note on Python/FWLite support:
//
// The ElectronMVAEstimatorRun2 is not only used in the full CMSSW framework
// via the AnyMVAEstimatorRun2Factory, but it is intended to also be used
// standalone in Python/FWLite. However, we want to avoid building dictionaries
// for the ElectronMVAEstimatorRun2 in this ElectronIdentification package,
// becase algorithms and data formats should not mix in CMSSW.
// That's why it has to be possible to create the dictionaries on the fly, for
// example by running this line in a Python script:
//
// ```Python
// ROOT.gInterpreter.Declare('#include "RecoEgamma/ElectronIdentification/interface/ElectronMVAEstimatorRun2.h"')
// ```
//
// To speed up the dictionary generation and avoid errors caused by conflicting
// C++ modules, we try to forwar declare as much as possible in
// ElectronMVAEstimatorRun2.h and AnyMVAEstimatorRun2Base.h.

class ElectronMVAEstimatorRun2 : public AnyMVAEstimatorRun2Base {
public:
  // Constructor
  ElectronMVAEstimatorRun2(const edm::ParameterSet& conf);
  // For use with FWLite/Python
  ElectronMVAEstimatorRun2(const std::string& mvaTag,
                           const std::string& mvaName,
                           int nCategories,
                           const std::string& variableDefinition,
                           const std::vector<std::string>& categoryCutStrings,
                           const std::vector<std::string>& weightFileNames,
                           bool debug = false);

  // Calculation of the MVA value
  float mvaValue(const reco::Candidate* candidate,
                 std::vector<float> const& auxVariables,
                 int& iCategory) const override;

  // for FWLite just passing rho
  float mvaValue(const reco::Candidate* candidate, float rho, int& iCategory) const {
    return mvaValue(candidate, std::vector<float>{rho}, iCategory);
  }

  int findCategory(const reco::Candidate* candidate) const override;

private:
  void init(const std::vector<std::string>& weightFileNames);

  int findCategory(reco::GsfElectron const& electron) const;

  std::vector<ThreadSafeFunctor<StringCutObjectSelector<reco::GsfElectron>>> categoryFunctions_;
  std::vector<int> nVariables_;

  // Data members
  std::vector<std::unique_ptr<const GBRForest>> gbrForests_;

  // There might be different variables for each category, so the variables
  // names vector is itself a vector of length nCategories
  std::vector<std::vector<int>> variables_;

  MVAVariableManager<reco::GsfElectron> mvaVarMngr_;
};

#endif
