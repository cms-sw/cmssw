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
  enum RunRange { RunAB, RunC, RunD };

  EcalIsolationCorrector(bool forElectrons);
  ~EcalIsolationCorrector(){};

#ifndef STANDALONE_ECALCORR
  // Global correction for ABCD together
  float correctForNoise(
      const reco::GsfElectron& e, bool isData = false, float intL_AB = 5.5, float intL_C = 6.7, float intL_D = 7.3);
  // Run dependent correction
  float correctForNoise(const reco::GsfElectron& e, int runNumber, bool isData = false);
  float correctForNoise(const reco::GsfElectron& e, const std::string& runName, bool isData = false);

  // Global correction for ABCD together
  float correctForHLTDefinition(
      const reco::GsfElectron& e, bool isData = false, float intL_AB = 5.5, float intL_C = 6.7, float intL_D = 7.3);
  // Run dependent correction
  float correctForHLTDefinition(const reco::GsfElectron& e, int runNumber, bool isData = false);
  float correctForHLTDefinition(const reco::GsfElectron& e, const std::string& runName, bool isData = false);

  // Global correction for ABCD together
  float correctForNoise(
      const reco::Photon& p, bool isData = false, float intL_AB = 5.5, float intL_C = 6.7, float intL_D = 7.3);
  // Run dependent correction
  float correctForNoise(const reco::Photon& p, int runNumber, bool isData = false);
  float correctForNoise(const reco::Photon& p, const std::string& runName, bool isData = false);

  // Global correction for ABCD together
  float correctForHLTDefinition(
      const reco::Photon& p, bool isData = false, float intL_AB = 5.5, float intL_C = 6.7, float intL_D = 7.3);
  // Run dependent correction
  float correctForHLTDefinition(const reco::Photon& p, int runNumber, bool isData = false);
  float correctForHLTDefinition(const reco::Photon& p, const std::string& runName, bool isData = false);

  // Global correction for ABCD together
  float correctForNoise(
      const pat::Electron& e, bool isData = false, float intL_AB = 5.5, float intL_C = 6.7, float intL_D = 7.3);
  // Run dependent correction
  float correctForNoise(const pat::Electron& e, int runNumber, bool isData = false);
  float correctForNoise(const pat::Electron& e, const std::string& runName, bool isData = false);

  // Global correction for ABCD together
  float correctForHLTDefinition(
      const pat::Electron& e, bool isData = false, float intL_AB = 5.5, float intL_C = 6.7, float intL_D = 7.3);
  // Run dependent correction
  float correctForHLTDefinition(const pat::Electron& e, int runNumber, bool isData = false);
  float correctForHLTDefinition(const pat::Electron& e, const std::string& runName, bool isData = false);

  // Global correction for ABCD together
  float correctForNoise(
      const pat::Photon& p, bool isData = false, float intL_AB = 5.5, float intL_C = 6.7, float intL_D = 7.3);
  // Run dependent correction
  float correctForNoise(const pat::Photon& p, int runNumber, bool isData = false);
  float correctForNoise(const pat::Photon& p, const std::string& runName, bool isData = false);

  // Global correction for ABCD together
  float correctForHLTDefinition(
      const pat::Photon& p, bool isData = false, float intL_AB = 5.5, float intL_C = 6.7, float intL_D = 7.3);
  // Run dependent correction
  float correctForHLTDefinition(const pat::Photon& p, int runNumber, bool isData = false);
  float correctForHLTDefinition(const pat::Photon& p, const std::string& runName, bool isData = false);
#else
  // Global correction for ABCD together
  float correctForNoise(
      float unCorrIso, bool isBarrel, bool isData = false, float intL_AB = 5.5, float intL_C = 6.7, float intL_D = 7.3);
  // Run dependent correction
  float correctForNoise(float unCorrIso, bool isBarrel, int runNumber, bool isData = false);
  float correctForNoise(float unCorrIso, bool isBarrel, std::string runName, bool isData = false);

  // Global correction for ABCD together
  float correctForHLTDefinition(float unCorrIso,
                                bool isBarrrel,
                                bool isData = false,
                                float intL_AB = 5.5,
                                float intL_C = 6.7,
                                float intL_D = 7.3);
  // Run dependent correction
  float correctForHLTDefinition(float unCorrIso, bool isBarrel, int runNumber, bool isData = false);
  float correctForHLTDefinition(float unCorrIso, bool isBarrel, std::string runName, bool isData = false);
#endif

protected:
  RunRange checkRunRange(int runNumber);
  float correctForNoise(float iso, bool isBarrel, RunRange runRange, bool isData);
  float correctForHLTDefinition(float iso, bool isBarrel, RunRange runRange);

private:
  bool isElectron_;
};
