#ifndef CALORIMETRYMANAGER_H
#define CALORIMETRYMANAGER_H

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimG4CMS/Calo/interface/CaloHitID.h"

// FastSimulation headers
#include "FastSimulation/Particle/interface/RawParticle.h"
#include "FastSimulation/Calorimetry/interface/HCALResponse.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FastSimulation/Utilities/interface/FamosDebug.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "FastSimulation/CaloHitMakers/interface/EcalHitMaker.h"
#include "FastSimulation/CaloHitMakers/interface/HcalHitMaker.h"
#include "FastSimulation/CaloHitMakers/interface/PreshowerHitMaker.h"

// For the uint32_t
//#include <boost/cstdint.hpp>
#include <map>
#include <algorithm>

class FSimEvent;
class FSimTrack;
class RawParticle;
class CaloGeometryHelper;
class Histos;
class HSParameters;
class RandomEngine;
class LandauFluctuationGenerator;
class GammaFunctionGenerator;
class MaterialEffects;
//Gflash
class GflashHadronShowerProfile;
class GflashPiKShowerProfile;
class GflashProtonShowerProfile;
class GflashAntiProtonShowerProfile;

class DQMStore;

namespace edm { 
  class ParameterSet;
}

class CalorimetryManager{

 public:
  CalorimetryManager();
  CalorimetryManager(FSimEvent* aSimEvent, 
		     const edm::ParameterSet& fastCalo,
		     const edm::ParameterSet& MuonECALPars,
		     const edm::ParameterSet& MuonHCALPars,
                     const edm::ParameterSet& fastGflash,
		     const RandomEngine* engine);
  ~CalorimetryManager();

  // Does the real job
  void reconstruct();

    // Return the address of the Calorimeter 
  CaloGeometryHelper * getCalorimeter() const {return myCalorimeter_;}

  // load container from edm::Event
  void loadFromEcalBarrel(edm::PCaloHitContainer & c) const;

  void loadFromEcalEndcap(edm::PCaloHitContainer & c) const;

  void loadFromHcal(edm::PCaloHitContainer & c) const;

  void loadFromPreshower(edm::PCaloHitContainer & c) const;

  void loadMuonSimTracks(edm::SimTrackContainer & m) const;

 private:
  // Simulation of electromagnetic showers in PS, ECAL, HCAL
  void EMShowerSimulation(const FSimTrack& myTrack);
  
  void reconstructHCAL(const FSimTrack& myTrack);

  void MuonMipSimulation(const FSimTrack & myTrack);
 
  /// Hadronic Shower Simulation
  void HDShowerSimulation(const FSimTrack& myTrack);

  // Read the parameters 
  void readParameters(const edm::ParameterSet& fastCalo);

  void updateECAL(const std::map<CaloHitID,float>& hitMap, int onEcal, int trackID=0, float corr=1.0); 
  void updateHCAL(const std::map<CaloHitID,float>& hitMap, int trackID=0, float corr=1.0); 
  void updatePreshower(const std::map<CaloHitID,float>& hitMap, int trackID=0, float corr=1.0); 
  
  void respCorr(double);

  void clean(); 

 private:

  FSimEvent* mySimEvent;
  CaloGeometryHelper* myCalorimeter_;

  Histos * myHistos;
  DQMStore * dbe;


  HCALResponse* myHDResponse_;
  HSParameters * myHSParameters_;

  std::vector<std::pair<CaloHitID,float> > EBMapping_;
  std::vector<std::pair<CaloHitID,float> > EEMapping_;
  std::vector<std::pair<CaloHitID,float> > HMapping_;
  std::vector<std::pair<CaloHitID,float> > ESMapping_;

  bool debug_;
  bool useDQM_;
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

  /// A few pointers to save time
  RawParticle myElec;
  RawParticle myPosi;
  RawParticle myPart;

  // Parameters 
  double pulledPadSurvivalProbability_;
  double crackPadSurvivalProbability_;
  double spotFraction_;
  //  double radiusFactor_;
  double radiusFactorEB_ , radiusFactorEE_;
  std::vector<double> radiusPreshowerCorrections_;
  double aTerm, bTerm;
  std::vector<double> mipValues_;
  int gridSize_;
  std::vector<double> theCoreIntervals_,theTailIntervals_;
  double RCFactor_,RTFactor_;
  //FR
  int optionHDSim_, hdGridSize_, hdSimMethod_;
  bool simulatePreshower_;
  //RF 

  // Famos Random Engine
  const RandomEngine* random;
  const LandauFluctuationGenerator* aLandauGenerator;
  GammaFunctionGenerator* aGammaGenerator;

  static std::vector<std::pair<int, float> > myZero_;

  // RespCorrP p, k_e(p), k_h(p) vectors  and evaluated for each p
  // ecorr and hcorr  
  std::vector<double> rsp;
  std::vector<double> p_knots;
  std::vector<double> k_e;
  std::vector<double> k_h;
  double ecorr;
  double hcorr;

  // Used to check if the calorimeters was initialized
  bool initialized_;

  std::vector<FSimTrack> muonSimTracks;
  MaterialEffects* theMuonEcalEffects; // material effects for muons in ECAL
  MaterialEffects* theMuonHcalEffects; // material effects for muons in HCAL


  // If set to true the simulation in ECAL would be done 1X0 by 1X0
  // this is slow but more adapted to detailed studies.
  // Otherwise roughty 5 steps are used.
  // This variable is transferred to EMShower
  bool bFixedLength_;

  //Gflash
  GflashHadronShowerProfile *theProfile;
  GflashPiKShowerProfile *thePiKProfile;
  GflashProtonShowerProfile *theProtonProfile;
  GflashAntiProtonShowerProfile *theAntiProtonProfile;
};
#endif
