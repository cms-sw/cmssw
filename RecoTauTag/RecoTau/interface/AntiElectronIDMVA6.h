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
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "CondFormats/GBRForest/interface/GBRForest.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "RecoTauTag/RecoTau/interface/PositionAtECalEntranceComputer.h"

#include "TMVA/Tools.h"
#include "TMVA/Reader.h"

#include "DataFormats/Math/interface/deltaR.h"

#include <vector>

namespace antiElecIDMVA6_blocks {
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
    float eSeedClusterOverPout = 0;
    float superClusterEtaWidth = 0;
    float superClusterPhiWidth = 0;
    float sigmaIEtaIEta5x5 = 0;
    float sigmaIPhiIPhi5x5 = 0;
    float showerCircularity = 0;
    float r9 = 0;
    float hgcalSigmaUU = 0;
    float hgcalSigmaVV = 0;
    float hgcalSigmaEE = 0;
    float hgcalSigmaPP = 0;
    float hgcalNLayers = 0;
    float hgcalFirstLayer = 0;
    float hgcalLastLayer = 0;
    float hgcalLayerEfrac10 = 0;
    float hgcalLayerEfrac90 = 0;
    float hgcalEcEnergyEE = 0;
    float hgcalEcEnergyFH = 0;
    float hgcalMeasuredDepth = 0;
    float hgcalExpectedDepth = 0;
    float hgcalExpectedSigma = 0;
    float hgcalDepthCompatibility = 0;
  };
}  // namespace antiElecIDMVA6_blocks

template <class TauType, class ElectronType>
class AntiElectronIDMVA6 {
public:
  typedef std::vector<ElectronType> ElectronCollection;
  typedef edm::Ref<ElectronCollection> ElectronRef;

  AntiElectronIDMVA6(const edm::ParameterSet&, edm::ConsumesCollector&&);
  ~AntiElectronIDMVA6();

  void beginEvent(const edm::Event&, const edm::EventSetup&);

  double mvaValue(const antiElecIDMVA6_blocks::TauVars& tauVars,
                  const antiElecIDMVA6_blocks::TauGammaVecs& tauGammaVecs,
                  const antiElecIDMVA6_blocks::ElecVars& elecVars);

  double mvaValue(const antiElecIDMVA6_blocks::TauVars& tauVars,
                  const antiElecIDMVA6_blocks::TauGammaMoms& tauGammaMoms,
                  const antiElecIDMVA6_blocks::ElecVars& elecVars);

  double mvaValuePhase2(const antiElecIDMVA6_blocks::TauVars& tauVars,
                        const antiElecIDMVA6_blocks::TauGammaMoms& tauGammaMoms,
                        const antiElecIDMVA6_blocks::ElecVars& elecVars);

  // this function can be called for all categories
  double mvaValue(const TauType& theTau, const ElectronRef& theEleRef);
  // this function can be called for category 1 only !!
  double mvaValue(const TauType& theTau);

  // overloaded method with explicit tau type to avoid partial imlementation of full class
  antiElecIDMVA6_blocks::TauVars getTauVarsTypeSpecific(const reco::PFTau& theTau);
  antiElecIDMVA6_blocks::TauVars getTauVarsTypeSpecific(const pat::Tau& theTau);
  antiElecIDMVA6_blocks::TauVars getTauVars(const TauType& theTau);
  antiElecIDMVA6_blocks::TauGammaVecs getTauGammaVecs(const TauType& theTau);
  antiElecIDMVA6_blocks::ElecVars getElecVars(const ElectronRef& theEleRef);
  // overloaded method with explicit electron type to avoid partial imlementation of full class
  void getElecVarsHGCalTypeSpecific(const reco::GsfElectronRef& theEleRef, antiElecIDMVA6_blocks::ElecVars& elecVars);
  void getElecVarsHGCalTypeSpecific(const pat::ElectronRef& theEleRef, antiElecIDMVA6_blocks::ElecVars& elecVars);

private:
  double dCrackEta(double eta);
  double minimum(double a, double b);
  double dCrackPhi(double phi, double eta);
  bool energyWeightedEtaAndPhiAtECal(
      const pat::Tau& theTau,
      float& eta,
      float& phi);  // MB: needed only for pat::Tau and called within pat::Tau specific method so also pat::Tau specific

  static constexpr float ecalBarrelEndcapEtaBorder_ = 1.479;
  static constexpr float ecalEndcapVFEndcapEtaBorder_ = 2.4;

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
  std::string mvaName_NoEleMatch_woGwoGSF_VFEC_;
  std::string mvaName_NoEleMatch_wGwoGSF_VFEC_;
  std::string mvaName_woGwGSF_VFEC_;
  std::string mvaName_wGwGSF_VFEC_;

  bool usePhiAtEcalEntranceExtrapolation_;

  std::vector<float> var_NoEleMatch_woGwoGSF_Barrel_;
  std::vector<float> var_NoEleMatch_wGwoGSF_Barrel_;
  std::vector<float> var_woGwGSF_Barrel_;
  std::vector<float> var_wGwGSF_Barrel_;
  std::vector<float> var_NoEleMatch_woGwoGSF_Endcap_;
  std::vector<float> var_NoEleMatch_wGwoGSF_Endcap_;
  std::vector<float> var_woGwGSF_Endcap_;
  std::vector<float> var_wGwGSF_Endcap_;
  std::vector<float> var_NoEleMatch_woGwoGSF_VFEndcap_;
  std::vector<float> var_NoEleMatch_wGwoGSF_VFEndcap_;
  std::vector<float> var_woGwGSF_VFEndcap_;
  std::vector<float> var_wGwGSF_VFEndcap_;

  const GBRForest* mva_NoEleMatch_woGwoGSF_BL_;
  const GBRForest* mva_NoEleMatch_wGwoGSF_BL_;
  const GBRForest* mva_woGwGSF_BL_;
  const GBRForest* mva_wGwGSF_BL_;
  const GBRForest* mva_NoEleMatch_woGwoGSF_EC_;
  const GBRForest* mva_NoEleMatch_wGwoGSF_EC_;
  const GBRForest* mva_woGwGSF_EC_;
  const GBRForest* mva_wGwGSF_EC_;
  const GBRForest* mva_NoEleMatch_woGwoGSF_VFEC_;
  const GBRForest* mva_NoEleMatch_wGwoGSF_VFEC_;
  const GBRForest* mva_woGwGSF_VFEC_;
  const GBRForest* mva_wGwGSF_VFEC_;

  std::vector<TFile*> inputFilesToDelete_;

  const bool isPhase2_;

  PositionAtECalEntranceComputer positionAtECalEntrance_;

  std::map<std::string, edm::EDGetTokenT<edm::ValueMap<float>>> electronIds_tokens_;
  std::map<std::string, edm::Handle<edm::ValueMap<float>>> electronIds_;

  const int verbosity_;
};

#endif
