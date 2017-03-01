#ifndef RecoMET_METPUSubtraction_PFMETAlgorithmMVA_h
#define RecoMET_METPUSubtraction_PFMETAlgorithmMVA_h

/** \class PFMETAlgorithmMVA
 *
 * MVA based algorithm for computing the particle-flow missing Et
 *
 * \authors Phil Harris, CERN
 *          Christian Veelken, LLR
 *
 */

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/EgammaObjects/interface/GBRForest.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "RecoMET/METPUSubtraction/interface/MvaMEtUtilities.h"

//#include <TMatrixD.h>
#include <Math/SMatrix.h>

#include <string>
#include <vector>
#include <ostream>

class PFMETAlgorithmMVA 
{
  
 public:

  PFMETAlgorithmMVA(const edm::ParameterSet& cfg);
  ~PFMETAlgorithmMVA();

  void initialize(const edm::EventSetup&);

  void setHasPhotons(bool hasPhotons) { hasPhotons_ = hasPhotons; }

  void setInput(const std::vector<reco::PUSubMETCandInfo>&,
		const std::vector<reco::PUSubMETCandInfo>&,
		const std::vector<reco::PUSubMETCandInfo>&,
		const std::vector<reco::Vertex::Point>&);

  void evaluateMVA();

  reco::Candidate::LorentzVector getMEt()    const { return mvaMEt_;    }
  const reco::METCovMatrix&    getMEtCov() const { return mvaMEtCov_; }

  double getU()     const { return mvaOutputU_;    }
  double getDPhi()  const { return mvaOutputDPhi_;  }
  double getCovU1() const { return mvaOutputCovU1_; }
  double getCovU2() const { return mvaOutputCovU2_; }
  
  void print(std::ostream&) const;

 private:
  const std::string updateVariableNames(std::string input);
  const GBRForest* loadMVAfromFile(const edm::FileInPath& inputFileName, const std::string& mvaName);
  const GBRForest* loadMVAfromDB(const edm::EventSetup& es, const std::string& mvaName);

  const float evaluateU();
  const float evaluateDPhi();
  const float evaluateCovU1();
  const float evaluateCovU2();

  MvaMEtUtilities utils_;
    
  std::string mvaNameU_;
  std::string mvaNameDPhi_;
  std::string mvaNameCovU1_;
  std::string mvaNameCovU2_;

  int    mvaType_;
  bool   hasPhotons_;

  double dZcut_;
  std::unique_ptr<float[]> createFloatVector(std::vector<std::string> variableNames);
  const float GetResponse(const GBRForest *Reader, std::vector<std::string> &variableNames);
  void computeMET();
  std::map<std::string, float> var_;


  float* mvaInputU_;
  float* mvaInputDPhi_;
  float* mvaInputCovU1_;
  float* mvaInputCovU2_;
  
  float mvaOutputU_;
  float mvaOutputDPhi_;
  float mvaOutputCovU1_;
  float mvaOutputCovU2_;

  std::vector<std::string> varForU_;
  std::vector<std::string> varForDPhi_;
  std::vector<std::string> varForCovU1_;
  std::vector<std::string> varForCovU2_;


  double sumLeptonPx_;
  double sumLeptonPy_;
  double chargedSumLeptonPx_;
  double chargedSumLeptonPy_;

  reco::Candidate::LorentzVector mvaMEt_;
  //TMatrixD mvaMEtCov_;
  reco::METCovMatrix mvaMEtCov_;

  const GBRForest* mvaReaderU_;
  const GBRForest* mvaReaderDPhi_;
  const GBRForest* mvaReaderCovU1_;
  const GBRForest* mvaReaderCovU2_;

  bool loadMVAfromDB_;

  edm::ParameterSet cfg_;
};
#endif
