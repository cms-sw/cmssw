//--------------------------------------------------------------------------------------------------
// AntiElectronIDMVA6
//
// Helper Class for applying MVA anti-electron discrimination
//
// Authors: F.Colombo, C.Veelken
//          M. Bluj (template version)
//--------------------------------------------------------------------------------------------------

#ifndef RECOTAUTAG_RECOTAU_AntiElectronIDMVA6_H
#define RECOTAUTAG_RECOTAU_AntiElectronIDMVA6_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "CondFormats/EgammaObjects/interface/GBRForest.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "RecoTauTag/RecoTau/interface/PositionAtECalEntranceComputer.h"

#include "TMVA/Tools.h"
#include "TMVA/Reader.h"

#include "DataFormats/Math/interface/deltaR.h"

#include <vector>

struct TauVars {
  float pt = 0;
  float etaAtEcalEntrance = 0;
  float phi = 0;
  float leadChargedPFCandPt = 0;
  float leadChargedPFCandEtaAtEcalEntrance = 0;
  float emFraction = 0;
  float leadPFChargedHadrHoP = 0;
  float leadPFChargedHadrEoP = 0;
  float visMassIn = 0;
  float dCrackEta = 0;
  float dCrackPhi = 0;
  float hasGsf = 0;
};
struct TauGammaVecs {
  std::vector<float> gammasdEtaInSigCone;
  std::vector<float> gammasdPhiInSigCone;
  std::vector<float> gammasPtInSigCone;
  std::vector<float> gammasdEtaOutSigCone;
  std::vector<float> gammasdPhiOutSigCone;
  std::vector<float> gammasPtOutSigCone;
};
struct TauGammaMoms {
  int signalPFGammaCandsIn = 0;
  int signalPFGammaCandsOut = 0;
  float gammaEtaMomIn = 0;
  float gammaEtaMomOut = 0;
  float gammaPhiMomIn = 0;
  float gammaPhiMomOut = 0;
  float gammaEnFracIn = 0;
  float gammaEnFracOut = 0;
};
struct ElecVars {
  float eta = 0;
  float phi = 0;
  float eTotOverPin = 0;
  float chi2NormGSF = 0;
  float chi2NormKF = 0;
  float gsfNumHits = 0;
  float kfNumHits = 0;
  float gsfTrackResol = 0;
  float gsfTracklnPt = 0;
  float pIn = 0;
  float pOut = 0;
  float eEcal = 0;
  float deltaEta = 0;
  float deltaPhi = 0;
  float mvaInSigmaEtaEta = 0;
  float mvaInHadEnergy = 0;
  float mvaInDeltaEta = 0;
};

template <class TauType, class ElectronType>
class AntiElectronIDMVA6 {
public:
  AntiElectronIDMVA6(const edm::ParameterSet&);
  ~AntiElectronIDMVA6();

  void beginEvent(const edm::Event&, const edm::EventSetup&);

  double MVAValue(const TauVars& tauVars, const TauGammaVecs& tauGammaVecs, const ElecVars& elecVars);

  double MVAValue(const TauVars& tauVars, const TauGammaMoms& tauGammaMoms, const ElecVars& elecVars);

  // this function can be called for all categories
  double MVAValue(const TauType& theTau, const ElectronType& theEle);
  // this function can be called for category 1 only !!
  double MVAValue(const TauType& theTau);

  // overloaded method with explicit tau type to avoid partial imlementation of full class
  TauVars getTauVarsTypeSpecific(const reco::PFTau& theTau);
  TauVars getTauVarsTypeSpecific(const pat::Tau& theTau);
  TauVars getTauVars(const TauType& theTau);
  TauGammaVecs getTauGammaVecs(const TauType& theTau);
  ElecVars getElecVars(const ElectronType& theEle);

  // track extrapolation to ECAL entrance (used to re-calculate variables that might not be available on miniAOD)
  bool atECalEntrance(const reco::Candidate* part, math::XYZPoint& pos);

private:
  double dCrackEta(double eta);
  double minimum(double a, double b);
  double dCrackPhi(double phi, double eta);

  bool isInitialized_;
  bool loadMVAfromDB_;
  edm::FileInPath inputFileName_;

  std::string mvaName_NoEleMatch_woGwoGSF_BL_;
  std::string mvaName_NoEleMatch_wGwoGSF_BL_;
  std::string mvaName_woGwGSF_BL_;
  std::string mvaName_wGwGSF_BL_;
  std::string mvaName_NoEleMatch_woGwoGSF_EC_;
  std::string mvaName_NoEleMatch_wGwoGSF_EC_;
  std::string mvaName_woGwGSF_EC_;
  std::string mvaName_wGwGSF_EC_;

  bool usePhiAtEcalEntranceExtrapolation_;

  float* Var_NoEleMatch_woGwoGSF_Barrel_;
  float* Var_NoEleMatch_wGwoGSF_Barrel_;
  float* Var_woGwGSF_Barrel_;
  float* Var_wGwGSF_Barrel_;
  float* Var_NoEleMatch_woGwoGSF_Endcap_;
  float* Var_NoEleMatch_wGwoGSF_Endcap_;
  float* Var_woGwGSF_Endcap_;
  float* Var_wGwGSF_Endcap_;

  const GBRForest* mva_NoEleMatch_woGwoGSF_BL_;
  const GBRForest* mva_NoEleMatch_wGwoGSF_BL_;
  const GBRForest* mva_woGwGSF_BL_;
  const GBRForest* mva_wGwGSF_BL_;
  const GBRForest* mva_NoEleMatch_woGwoGSF_EC_;
  const GBRForest* mva_NoEleMatch_wGwoGSF_EC_;
  const GBRForest* mva_woGwGSF_EC_;
  const GBRForest* mva_wGwGSF_EC_;

  std::vector<TFile*> inputFilesToDelete_;

  PositionAtECalEntranceComputer positionAtECalEntrance_;

  int verbosity_;
};

#endif
