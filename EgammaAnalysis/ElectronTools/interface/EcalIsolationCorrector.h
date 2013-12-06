//
// to use this code outside of CMSSW
// set this definition
//

//#define STANDALONE_ECALCORR
#ifndef STANDALONE_ECALCORR
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#endif

#include <string>
#include <iostream>

class EcalIsolationCorrector {
 public:

  enum RunRange {RunAB, RunC, RunD};

  EcalIsolationCorrector(bool forElectrons);
  ~EcalIsolationCorrector() {};

#ifndef STANDALONE_ECALCORR
  // Global correction for ABCD together
  float correctForNoise(reco::GsfElectron e, bool isData=false, float intL_AB=5.5, float intL_C=6.7, float intL_D=7.3);
  // Run dependent correction
  float correctForNoise(reco::GsfElectron e, int runNumber, bool isData=false);
  float correctForNoise(reco::GsfElectron e, std::string runName, bool isData=false);

  // Global correction for ABCD together
  float correctForHLTDefinition(reco::GsfElectron e, bool isData=false, float intL_AB=5.5, float intL_C=6.7, float intL_D=7.3);
  // Run dependent correction
  float correctForHLTDefinition(reco::GsfElectron e, int runNumber, bool isData=false);
  float correctForHLTDefinition(reco::GsfElectron e, std::string runName, bool isData=false);

  // Global correction for ABCD together
  float correctForNoise(reco::Photon p, bool isData=false, float intL_AB=5.5, float intL_C=6.7, float intL_D=7.3);
  // Run dependent correction
  float correctForNoise(reco::Photon p, int runNumber, bool isData=false);
  float correctForNoise(reco::Photon p, std::string runName, bool isData=false);

  // Global correction for ABCD together
  float correctForHLTDefinition(reco::Photon p, bool isData=false, float intL_AB=5.5, float intL_C=6.7, float intL_D=7.3);
  // Run dependent correction
  float correctForHLTDefinition(reco::Photon p, int runNumber, bool isData=false);
  float correctForHLTDefinition(reco::Photon p, std::string runName, bool isData=false);

  // Global correction for ABCD together
  float correctForNoise(pat::Electron e, bool isData=false, float intL_AB=5.5, float intL_C=6.7, float intL_D=7.3);
  // Run dependent correction
  float correctForNoise(pat::Electron e, int runNumber, bool isData=false);
  float correctForNoise(pat::Electron e, std::string runName, bool isData=false);

  // Global correction for ABCD together
  float correctForHLTDefinition(pat::Electron e, bool isData=false, float intL_AB=5.5, float intL_C=6.7, float intL_D=7.3);
  // Run dependent correction
  float correctForHLTDefinition(pat::Electron e, int runNumber, bool isData=false);
  float correctForHLTDefinition(pat::Electron e, std::string runName, bool isData=false);

  // Global correction for ABCD together
  float correctForNoise(pat::Photon p, bool isData=false, float intL_AB=5.5, float intL_C=6.7, float intL_D=7.3);
  // Run dependent correction
  float correctForNoise(pat::Photon p, int runNumber, bool isData=false);
  float correctForNoise(pat::Photon p, std::string runName, bool isData=false);

  // Global correction for ABCD together
  float correctForHLTDefinition(pat::Photon p, bool isData=false, float intL_AB=5.5, float intL_C=6.7, float intL_D=7.3);
  // Run dependent correction
  float correctForHLTDefinition(pat::Photon p, int runNumber, bool isData=false);
  float correctForHLTDefinition(pat::Photon p, std::string runName, bool isData=false);
#else
  // Global correction for ABCD together
  float correctForNoise(float unCorrIso, bool isBarrel, bool isData=false, float intL_AB=5.5, float intL_C=6.7, float intL_D=7.3);
  // Run dependent correction
  float correctForNoise(float unCorrIso, bool isBarrel, int runNumber, bool isData=false);
  float correctForNoise(float unCorrIso, bool isBarrel, std::string runName, bool isData=false);

  // Global correction for ABCD together
  float correctForHLTDefinition(float unCorrIso, bool isBarrrel, bool isData=false, float intL_AB=5.5, float intL_C=6.7, float intL_D=7.3);
  // Run dependent correction
  float correctForHLTDefinition(float unCorrIso, bool isBarrel, int runNumber, bool isData=false);
  float correctForHLTDefinition(float unCorrIso, bool isBarrel, std::string runName, bool isData=false);
#endif

 protected:
  RunRange checkRunRange(int runNumber);
  float correctForNoise(float iso, bool isBarrel, RunRange runRange, bool isData);
  float correctForHLTDefinition(float iso, bool isBarrel, RunRange runRange);

 private:
  bool isElectron_;
};
