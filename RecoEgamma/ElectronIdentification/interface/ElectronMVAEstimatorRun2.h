#ifndef RecoEgamma_ElectronIdentification_ElectronMVAEstimatorRun2_H
#define RecoEgamma_ElectronIdentification_ElectronMVAEstimatorRun2_H

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "RecoEgamma/EgammaTools/interface/AnyMVAEstimatorRun2Base.h"
#include "CommonTools/MVAUtils/interface/GBRForestTools.h"
#include "RecoEgamma/EgammaTools/interface/MVAVariableManager.h"
#include "RecoEgamma/EgammaTools/interface/ThreadSafeStringCut.h"

#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include <TMath.h>

class ElectronMVAEstimatorRun2 : public AnyMVAEstimatorRun2Base {

 public:

  // Constructor and destructor
  ElectronMVAEstimatorRun2(const edm::ParameterSet& conf);
  ~ElectronMVAEstimatorRun2() override {};
  // For use with FWLite/Python
  ElectronMVAEstimatorRun2(const std::string& mvaTag,
                           const std::string& mvaName,
                           int nCategories,
                           const std::string& variableDefinition,
                           const std::vector<std::string>& categoryCutStrings,
                           const std::vector<std::string> &weightFileNames,
                           bool debug=false );

  // For use with FWLite/Python
  static std::vector<float> getExtraVars(double rho)
  {
      return std::vector<float>{static_cast<float>(rho)};
  }

  // Calculation of the MVA value
  float mvaValue( const reco::Candidate* candidate, std::vector<float> const& auxVariables, int &iCategory) const override;

  int findCategory( const reco::Candidate* candidate) const override;

 private:

  void init(const std::vector<std::string> &weightFileNames);

  int findCategory(reco::GsfElectron const& electron) const;

  std::vector<ThreadSafeStringCut<StringCutObjectSelector<reco::GsfElectron>, reco::GsfElectron>> categoryFunctions_;
  std::vector<int> nVariables_;

  // Data members
  std::vector< std::unique_ptr<const GBRForest> > gbrForests_;


  // There might be different variables for each category, so the variables
  // names vector is itself a vector of length nCategories
  std::vector<std::vector<int>> variables_;

  MVAVariableManager<reco::GsfElectron> mvaVarMngr_;

};

#endif
