#ifndef FastSimulation_Calorimetry_CalorimetryManager_h
#define FastSimulation_Calorimetry_CalorimetryManager_h

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimG4CMS/Calo/interface/CaloHitID.h"

// FastSimulation headers
#include "FastSimulation/Calorimetry/interface/HCALResponse.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FastSimulation/Utilities/interface/FamosDebug.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "FastSimulation/CaloHitMakers/interface/EcalHitMaker.h"
#include "FastSimulation/CaloHitMakers/interface/HcalHitMaker.h"
#include "FastSimulation/CaloHitMakers/interface/PreshowerHitMaker.h"
#include "FastSimulation/CalorimeterProperties/interface/CalorimetryConsumer.h"
#include "FastSimulation/Calorimetry/interface/KKCorrectionFactors.h"

#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"

#include <map>
#include <algorithm>
#include <utility>

class FSimEvent;
class FSimTrack;
class CaloGeometryHelper;
class Histos;
class HSParameters;
class LandauFluctuationGenerator;
class GammaFunctionGenerator;
class MaterialEffects;
class RandomEngineAndDistribution;
//Gflash
class GflashHadronShowerProfile;
class GflashPiKShowerProfile;
class GflashProtonShowerProfile;
class GflashAntiProtonShowerProfile;
// FastHFshowerLibrary
class FastHFShowerLibrary;

struct CaloProductContainer {
  CaloProductContainer() :
    hitsEB(std::make_unique<edm::PCaloHitContainer>()),
    hitsEE(std::make_unique<edm::PCaloHitContainer>()),
    hitsES(std::make_unique<edm::PCaloHitContainer>()),
    hitsHCAL(std::make_unique<edm::PCaloHitContainer>()),
    tracksMuon(std::make_unique<edm::SimTrackContainer>())
  {}

  std::unique_ptr<edm::PCaloHitContainer> hitsEB;
  std::unique_ptr<edm::PCaloHitContainer> hitsEE;
  std::unique_ptr<edm::PCaloHitContainer> hitsES;
  std::unique_ptr<edm::PCaloHitContainer> hitsHCAL;
  std::unique_ptr<edm::SimTrackContainer> tracksMuon;
};

class CalorimetryManager {
public:
  CalorimetryManager();
  CalorimetryManager(const edm::ParameterSet& fastCalo,
                     const edm::ParameterSet& MuonECALPars,
                     const edm::ParameterSet& MuonHCALPars,
                     const edm::ParameterSet& fastGflash,
                     double magneticFieldOrigin,
                     const edm::EventSetup& iSetup,
                     const CalorimetryConsumer& iConsumer);
  ~CalorimetryManager();

  // Does the real job
  void reconstructTrack(const FSimTrack& myTrack, RandomEngineAndDistribution const*, CaloProductContainer& container);

  // Return the address of the Calorimeter
  CaloGeometryHelper* getCalorimeter() const { return myCalorimeter_.get(); }

  // Return the address of the FastHFShowerLibrary
  FastHFShowerLibrary* getHFShowerLibrary() const { return theHFShowerLibrary_.get(); }

private:
  // Simulation of electromagnetic showers in PS, ECAL, HCAL
  void EMShowerSimulation(const FSimTrack& myTrack, RandomEngineAndDistribution const*, CaloProductContainer& container);

  void reconstructHCAL(const FSimTrack& myTrack, RandomEngineAndDistribution const*, CaloProductContainer& container);

  void MuonMipSimulation(const FSimTrack& myTrack, RandomEngineAndDistribution const*, CaloProductContainer& container);

  /// Hadronic Shower Simulation
  void HDShowerSimulation(const FSimTrack& myTrack, RandomEngineAndDistribution const*, CaloProductContainer& container);

  // Read the parameters
  void readParameters(const edm::ParameterSet& fastCalo);

  void updateECAL(const std::map<CaloHitID, float>& hitMap, int onEcal, int trackID, CaloProductContainer& container, float corr = 1.0) const;
  void updateHCAL(const std::map<CaloHitID, float>& hitMap,
                  bool usedShowerLibrary,
                  int trackID,
                  CaloProductContainer& container,
                  float corr = 1.0,
                  const std::vector<double>& hfcorrEm = {},
                  const std::vector<double>& hfcorrHad = {}) const;
  void updatePreshower(const std::map<CaloHitID, float>& hitMap, int trackID, CaloProductContainer& container, float corr = 1.0) const;
  void updateMuon(const FSimTrack& track, CaloProductContainer& container) const;

  std::pair<double, double> respCorr(double) const;

private:
  std::unique_ptr<CaloGeometryHelper> myCalorimeter_;

  std::unique_ptr<HCALResponse> myHDResponse_;
  std::unique_ptr<HSParameters> myHSParameters_;

  bool debug_;
  std::vector<unsigned int> evtsToDebug_;

  bool unfoldedMode_;

  //Digitizer
  bool EcalDigitizer_;
  bool HcalDigitizer_;
  std::vector<double> samplingHBHE_;
  std::vector<double> samplingHF_;
  std::vector<double> samplingHO_;
  int ietaShiftHB_, ietaShiftHE_, ietaShiftHO_, ietaShiftHF_;
  std::vector<double> timeShiftHB_;
  std::vector<double> timeShiftHE_;
  std::vector<double> timeShiftHF_;
  std::vector<double> timeShiftHO_;

  // Parameters
  double pulledPadSurvivalProbability_;
  double crackPadSurvivalProbability_;
  double spotFraction_;
  double radiusFactorEB_, radiusFactorEE_;
  std::vector<double> radiusPreshowerCorrections_;
  double aTerm_, bTerm_;
  std::vector<double> mipValues_;
  int gridSize_;
  std::vector<double> theCoreIntervals_, theTailIntervals_;
  double RCFactor_, RTFactor_;
  //FR
  int optionHDSim_, hdGridSize_, hdSimMethod_;
  bool simulatePreshower_;
  //RF

  std::unique_ptr<LandauFluctuationGenerator> aLandauGenerator_;
  std::unique_ptr<GammaFunctionGenerator> aGammaGenerator_;

  static std::vector<std::pair<int, float> > myZero_;

  // RespCorrP p, k_e(p), k_h(p) vectors  and evaluated for each p
  // ecorr and hcorr
  std::vector<double> rsp_;
  std::vector<double> p_knots_;
  std::vector<double> k_e_;
  std::vector<double> k_h_;

  std::unique_ptr<MaterialEffects> theMuonEcalEffects_;  // material effects for muons in ECAL
  std::unique_ptr<MaterialEffects> theMuonHcalEffects_;  // material effects for muons in HCAL

  // If set to true the simulation in ECAL would be done 1X0 by 1X0
  // this is slow but more adapted to detailed studies.
  // Otherwise roughly 5 steps are used.
  // This variable is transferred to EMShower
  bool bFixedLength_;

  //Gflash
  std::unique_ptr<GflashPiKShowerProfile> thePiKProfile_;
  std::unique_ptr<GflashProtonShowerProfile> theProtonProfile_;
  std::unique_ptr<GflashAntiProtonShowerProfile> theAntiProtonProfile_;

  // HFShowerLibrary
  bool useShowerLibrary_;
  bool useCorrectionSL_;
  std::unique_ptr<FastHFShowerLibrary> theHFShowerLibrary_;

  std::unique_ptr<KKCorrectionFactors> ecalCorrection_;
  const HepPDT::ParticleDataTable* pdt_;
};
#endif
