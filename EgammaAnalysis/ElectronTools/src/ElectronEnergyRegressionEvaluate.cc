#include "EgammaAnalysis/ElectronTools/interface/ElectronEnergyRegressionEvaluate.h"
#include <cmath>
#include <cassert>

#ifndef STANDALONE
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#endif

ElectronEnergyRegressionEvaluate::ElectronEnergyRegressionEvaluate()
    : fIsInitialized(kFALSE),
      fVersionType(kNoTrkVar),
      forestCorrection_eb(nullptr),
      forestCorrection_ee(nullptr),
      forestUncertainty_eb(nullptr),
      forestUncertainty_ee(nullptr) {}

ElectronEnergyRegressionEvaluate::~ElectronEnergyRegressionEvaluate() {}
// Destructor does nothing

void ElectronEnergyRegressionEvaluate::initialize(std::string weightsFile,
                                                  ElectronEnergyRegressionEvaluate::ElectronEnergyRegressionType type) {
  // Loading forest object according to different versions
  TFile file(edm::FileInPath(weightsFile.c_str()).fullPath().c_str());

  forestCorrection_eb = (GBRForest *)file.Get("EBCorrection");
  forestCorrection_ee = (GBRForest *)file.Get("EECorrection");
  forestUncertainty_eb = (GBRForest *)file.Get("EBUncertainty");
  forestUncertainty_ee = (GBRForest *)file.Get("EEUncertainty");

  // Just checking
  assert(forestCorrection_eb);
  assert(forestCorrection_ee);
  assert(forestUncertainty_eb);
  assert(forestUncertainty_ee);

  // Updating type and marking as initialized
  fVersionType = type;
  fIsInitialized = kTRUE;
}

#ifndef STANDALONE
double ElectronEnergyRegressionEvaluate::calculateRegressionEnergy(
    const reco::GsfElectron *ele, SuperClusterHelper &mySCHelper, double rho, double nvertices, bool printDebug) {
  if (!fIsInitialized) {
    std::cout << "Error: Electron Energy Regression has not been initialized yet. return 0. \n";
    return 0;
  }

  if (printDebug) {
    std::cout << "Regression Type: " << fVersionType << std::endl;
    std::cout << "Electron : " << ele->pt() << " " << ele->eta() << " " << ele->phi() << "\n";
  }

  if (fVersionType == kNoTrkVar) {
    return regressionValueNoTrkVar(mySCHelper.rawEnergy(),
                                   mySCHelper.eta(),
                                   mySCHelper.phi(),
                                   mySCHelper.r9(),
                                   mySCHelper.etaWidth(),
                                   mySCHelper.phiWidth(),
                                   mySCHelper.clustersSize(),
                                   mySCHelper.hadronicOverEm(),
                                   rho,
                                   nvertices,
                                   mySCHelper.seedEta(),
                                   mySCHelper.seedPhi(),
                                   mySCHelper.seedEnergy(),
                                   mySCHelper.e3x3(),
                                   mySCHelper.e5x5(),
                                   mySCHelper.sigmaIetaIeta(),
                                   mySCHelper.spp(),
                                   mySCHelper.sep(),
                                   mySCHelper.eMax(),
                                   mySCHelper.e2nd(),
                                   mySCHelper.eTop(),
                                   mySCHelper.eBottom(),
                                   mySCHelper.eLeft(),
                                   mySCHelper.eRight(),
                                   mySCHelper.e2x5Max(),
                                   mySCHelper.e2x5Top(),
                                   mySCHelper.e2x5Bottom(),
                                   mySCHelper.e2x5Left(),
                                   mySCHelper.e2x5Right(),
                                   mySCHelper.ietaSeed(),
                                   mySCHelper.iphiSeed(),
                                   mySCHelper.etaCrySeed(),
                                   mySCHelper.phiCrySeed(),
                                   mySCHelper.preshowerEnergyOverRaw(),
                                   printDebug);
  }
  if (fVersionType == kWithSubCluVar) {
    return regressionValueWithSubClusters(mySCHelper.rawEnergy(),
                                          mySCHelper.eta(),
                                          mySCHelper.phi(),
                                          mySCHelper.r9(),
                                          mySCHelper.etaWidth(),
                                          mySCHelper.phiWidth(),
                                          mySCHelper.clustersSize(),
                                          mySCHelper.hadronicOverEm(),
                                          rho,
                                          nvertices,
                                          mySCHelper.seedEta(),
                                          mySCHelper.seedPhi(),
                                          mySCHelper.seedEnergy(),
                                          mySCHelper.e3x3(),
                                          mySCHelper.e5x5(),
                                          mySCHelper.sigmaIetaIeta(),
                                          mySCHelper.spp(),
                                          mySCHelper.sep(),
                                          mySCHelper.eMax(),
                                          mySCHelper.e2nd(),
                                          mySCHelper.eTop(),
                                          mySCHelper.eBottom(),
                                          mySCHelper.eLeft(),
                                          mySCHelper.eRight(),
                                          mySCHelper.e2x5Max(),
                                          mySCHelper.e2x5Top(),
                                          mySCHelper.e2x5Bottom(),
                                          mySCHelper.e2x5Left(),
                                          mySCHelper.e2x5Right(),
                                          mySCHelper.ietaSeed(),
                                          mySCHelper.iphiSeed(),
                                          mySCHelper.etaCrySeed(),
                                          mySCHelper.phiCrySeed(),
                                          mySCHelper.preshowerEnergyOverRaw(),
                                          ele->ecalDrivenSeed(),
                                          ele->isEBEtaGap(),
                                          ele->isEBPhiGap(),
                                          ele->isEEDeeGap(),
                                          mySCHelper.eSubClusters(),
                                          mySCHelper.subClusterEnergy(1),
                                          mySCHelper.subClusterEta(1),
                                          mySCHelper.subClusterPhi(1),
                                          mySCHelper.subClusterEmax(1),
                                          mySCHelper.subClusterE3x3(1),
                                          mySCHelper.subClusterEnergy(2),
                                          mySCHelper.subClusterEta(2),
                                          mySCHelper.subClusterPhi(2),
                                          mySCHelper.subClusterEmax(2),
                                          mySCHelper.subClusterE3x3(2),
                                          mySCHelper.subClusterEnergy(3),
                                          mySCHelper.subClusterEta(3),
                                          mySCHelper.subClusterPhi(3),
                                          mySCHelper.subClusterEmax(3),
                                          mySCHelper.subClusterE3x3(3),
                                          mySCHelper.nPreshowerClusters(),
                                          mySCHelper.eESClusters(),
                                          mySCHelper.esClusterEnergy(0),
                                          mySCHelper.esClusterEta(0),
                                          mySCHelper.esClusterPhi(0),
                                          mySCHelper.esClusterEnergy(1),
                                          mySCHelper.esClusterEta(1),
                                          mySCHelper.esClusterPhi(1),
                                          mySCHelper.esClusterEnergy(2),
                                          mySCHelper.esClusterEta(2),
                                          mySCHelper.esClusterPhi(2),
                                          ele->isEB(),
                                          printDebug);
  } else if (fVersionType == kNoTrkVarV1) {
    return regressionValueNoTrkVarV1(mySCHelper.rawEnergy(),
                                     mySCHelper.eta(),
                                     mySCHelper.phi(),
                                     mySCHelper.r9(),
                                     mySCHelper.etaWidth(),
                                     mySCHelper.phiWidth(),
                                     mySCHelper.clustersSize(),
                                     mySCHelper.hadronicOverEm(),
                                     rho,
                                     nvertices,
                                     mySCHelper.seedEta(),
                                     mySCHelper.seedPhi(),
                                     mySCHelper.seedEnergy(),
                                     mySCHelper.e3x3(),
                                     mySCHelper.e5x5(),
                                     mySCHelper.sigmaIetaIeta(),
                                     mySCHelper.spp(),
                                     mySCHelper.sep(),
                                     mySCHelper.eMax(),
                                     mySCHelper.e2nd(),
                                     mySCHelper.eTop(),
                                     mySCHelper.eBottom(),
                                     mySCHelper.eLeft(),
                                     mySCHelper.eRight(),
                                     mySCHelper.e2x5Max(),
                                     mySCHelper.e2x5Top(),
                                     mySCHelper.e2x5Bottom(),
                                     mySCHelper.e2x5Left(),
                                     mySCHelper.e2x5Right(),
                                     mySCHelper.ietaSeed(),
                                     mySCHelper.iphiSeed(),
                                     mySCHelper.etaCrySeed(),
                                     mySCHelper.phiCrySeed(),
                                     mySCHelper.preshowerEnergyOverRaw(),
                                     ele->ecalDrivenSeed(),
                                     printDebug);
  } else if (fVersionType == kWithTrkVarV1) {
    return regressionValueWithTrkVarV1(mySCHelper.rawEnergy(),
                                       mySCHelper.eta(),
                                       mySCHelper.phi(),
                                       mySCHelper.r9(),
                                       mySCHelper.etaWidth(),
                                       mySCHelper.phiWidth(),
                                       mySCHelper.clustersSize(),
                                       mySCHelper.hadronicOverEm(),
                                       rho,
                                       nvertices,
                                       mySCHelper.seedEta(),
                                       mySCHelper.seedPhi(),
                                       mySCHelper.seedEnergy(),
                                       mySCHelper.e3x3(),
                                       mySCHelper.e5x5(),
                                       mySCHelper.sigmaIetaIeta(),
                                       mySCHelper.spp(),
                                       mySCHelper.sep(),
                                       mySCHelper.eMax(),
                                       mySCHelper.e2nd(),
                                       mySCHelper.eTop(),
                                       mySCHelper.eBottom(),
                                       mySCHelper.eLeft(),
                                       mySCHelper.eRight(),
                                       mySCHelper.e2x5Max(),
                                       mySCHelper.e2x5Top(),
                                       mySCHelper.e2x5Bottom(),
                                       mySCHelper.e2x5Left(),
                                       mySCHelper.e2x5Right(),
                                       mySCHelper.ietaSeed(),
                                       mySCHelper.iphiSeed(),
                                       mySCHelper.etaCrySeed(),
                                       mySCHelper.phiCrySeed(),
                                       mySCHelper.preshowerEnergyOverRaw(),
                                       ele->ecalDrivenSeed(),
                                       ele->trackMomentumAtVtx().R(),
                                       fmax(ele->fbrem(), -1.0),
                                       ele->charge(),
                                       fmin(ele->eSuperClusterOverP(), 20.0),
                                       ele->trackMomentumError(),
                                       ele->correctedEcalEnergyError(),
                                       ele->classification(),
                                       printDebug);
  } else if (fVersionType == kWithTrkVarV2) {
    return regressionValueWithTrkVarV2(
        mySCHelper.rawEnergy(),
        mySCHelper.eta(),
        mySCHelper.phi(),
        mySCHelper.r9(),
        mySCHelper.etaWidth(),
        mySCHelper.phiWidth(),
        mySCHelper.clustersSize(),
        mySCHelper.hadronicOverEm(),
        rho,
        nvertices,
        mySCHelper.seedEta(),
        mySCHelper.seedPhi(),
        mySCHelper.seedEnergy(),
        mySCHelper.e3x3(),
        mySCHelper.e5x5(),
        mySCHelper.sigmaIetaIeta(),
        mySCHelper.spp(),
        mySCHelper.sep(),
        mySCHelper.eMax(),
        mySCHelper.e2nd(),
        mySCHelper.eTop(),
        mySCHelper.eBottom(),
        mySCHelper.eLeft(),
        mySCHelper.eRight(),
        mySCHelper.e2x5Max(),
        mySCHelper.e2x5Top(),
        mySCHelper.e2x5Bottom(),
        mySCHelper.e2x5Left(),
        mySCHelper.e2x5Right(),
        mySCHelper.ietaSeed(),
        mySCHelper.iphiSeed(),
        mySCHelper.etaCrySeed(),
        mySCHelper.phiCrySeed(),
        mySCHelper.preshowerEnergyOverRaw(),
        ele->ecalDrivenSeed(),
        ele->trackMomentumAtVtx().R(),
        fmax(ele->fbrem(), -1.0),
        ele->charge(),
        fmin(ele->eSuperClusterOverP(), 20.0),
        ele->trackMomentumError(),
        ele->correctedEcalEnergyError(),
        ele->classification(),
        fmin(std::abs(ele->deltaEtaSuperClusterTrackAtVtx()), 0.6),
        ele->deltaPhiSuperClusterTrackAtVtx(),
        ele->deltaEtaSeedClusterTrackAtCalo(),
        ele->deltaPhiSeedClusterTrackAtCalo(),
        ele->gsfTrack()->chi2() / ele->gsfTrack()->ndof(),
        (ele->closestCtfTrackRef().isNonnull() ? ele->closestCtfTrackRef()->hitPattern().trackerLayersWithMeasurement()
                                               : -1),
        fmin(ele->eEleClusterOverPout(), 20.0),
        printDebug);
  } else {
    std::cout << "Warning: Electron Regression Type " << fVersionType
              << " is not supported. Reverting to default electron momentum.\n";
    return ele->p();
  }
}

double ElectronEnergyRegressionEvaluate::calculateRegressionEnergyUncertainty(
    const reco::GsfElectron *ele, SuperClusterHelper &mySCHelper, double rho, double nvertices, bool printDebug) {
  if (!fIsInitialized) {
    std::cout << "Error: Electron Energy Regression has not been initialized yet. return 0. \n";
    return 0;
  }

  if (printDebug) {
    std::cout << "Regression Type: " << fVersionType << std::endl;
    std::cout << "Electron : " << ele->pt() << " " << ele->eta() << " " << ele->phi() << "\n";
  }

  if (fVersionType == kNoTrkVar) {
    return regressionUncertaintyNoTrkVar(mySCHelper.rawEnergy(),
                                         mySCHelper.eta(),
                                         mySCHelper.phi(),
                                         mySCHelper.r9(),
                                         mySCHelper.etaWidth(),
                                         mySCHelper.phiWidth(),
                                         mySCHelper.clustersSize(),
                                         mySCHelper.hadronicOverEm(),
                                         rho,
                                         nvertices,
                                         mySCHelper.seedEta(),
                                         mySCHelper.seedPhi(),
                                         mySCHelper.seedEnergy(),
                                         mySCHelper.e3x3(),
                                         mySCHelper.e5x5(),
                                         mySCHelper.sigmaIetaIeta(),
                                         mySCHelper.spp(),
                                         mySCHelper.sep(),
                                         mySCHelper.eMax(),
                                         mySCHelper.e2nd(),
                                         mySCHelper.eTop(),
                                         mySCHelper.eBottom(),
                                         mySCHelper.eLeft(),
                                         mySCHelper.eRight(),
                                         mySCHelper.e2x5Max(),
                                         mySCHelper.e2x5Top(),
                                         mySCHelper.e2x5Bottom(),
                                         mySCHelper.e2x5Left(),
                                         mySCHelper.e2x5Right(),
                                         mySCHelper.ietaSeed(),
                                         mySCHelper.iphiSeed(),
                                         mySCHelper.etaCrySeed(),
                                         mySCHelper.phiCrySeed(),
                                         mySCHelper.preshowerEnergyOverRaw(),
                                         printDebug);
  }
  if (fVersionType == kWithSubCluVar) {
    return regressionUncertaintyWithSubClusters(mySCHelper.rawEnergy(),
                                                mySCHelper.eta(),
                                                mySCHelper.phi(),
                                                mySCHelper.r9(),
                                                mySCHelper.etaWidth(),
                                                mySCHelper.phiWidth(),
                                                mySCHelper.clustersSize(),
                                                mySCHelper.hadronicOverEm(),
                                                rho,
                                                nvertices,
                                                mySCHelper.seedEta(),
                                                mySCHelper.seedPhi(),
                                                mySCHelper.seedEnergy(),
                                                mySCHelper.e3x3(),
                                                mySCHelper.e5x5(),
                                                mySCHelper.sigmaIetaIeta(),
                                                mySCHelper.spp(),
                                                mySCHelper.sep(),
                                                mySCHelper.eMax(),
                                                mySCHelper.e2nd(),
                                                mySCHelper.eTop(),
                                                mySCHelper.eBottom(),
                                                mySCHelper.eLeft(),
                                                mySCHelper.eRight(),
                                                mySCHelper.e2x5Max(),
                                                mySCHelper.e2x5Top(),
                                                mySCHelper.e2x5Bottom(),
                                                mySCHelper.e2x5Left(),
                                                mySCHelper.e2x5Right(),
                                                mySCHelper.ietaSeed(),
                                                mySCHelper.iphiSeed(),
                                                mySCHelper.etaCrySeed(),
                                                mySCHelper.phiCrySeed(),
                                                mySCHelper.preshowerEnergyOverRaw(),
                                                ele->ecalDrivenSeed(),
                                                ele->isEBEtaGap(),
                                                ele->isEBPhiGap(),
                                                ele->isEEDeeGap(),
                                                mySCHelper.eSubClusters(),
                                                mySCHelper.subClusterEnergy(1),
                                                mySCHelper.subClusterEta(1),
                                                mySCHelper.subClusterPhi(1),
                                                mySCHelper.subClusterEmax(1),
                                                mySCHelper.subClusterE3x3(1),
                                                mySCHelper.subClusterEnergy(2),
                                                mySCHelper.subClusterEta(2),
                                                mySCHelper.subClusterPhi(2),
                                                mySCHelper.subClusterEmax(2),
                                                mySCHelper.subClusterE3x3(2),
                                                mySCHelper.subClusterEnergy(3),
                                                mySCHelper.subClusterEta(3),
                                                mySCHelper.subClusterPhi(3),
                                                mySCHelper.subClusterEmax(3),
                                                mySCHelper.subClusterE3x3(3),
                                                mySCHelper.nPreshowerClusters(),
                                                mySCHelper.eESClusters(),
                                                mySCHelper.esClusterEnergy(0),
                                                mySCHelper.esClusterEta(0),
                                                mySCHelper.esClusterPhi(0),
                                                mySCHelper.esClusterEnergy(1),
                                                mySCHelper.esClusterEta(1),
                                                mySCHelper.esClusterPhi(1),
                                                mySCHelper.esClusterEnergy(2),
                                                mySCHelper.esClusterEta(2),
                                                mySCHelper.esClusterPhi(2),
                                                ele->isEB(),
                                                printDebug);
  } else if (fVersionType == kNoTrkVarV1) {
    return regressionUncertaintyNoTrkVarV1(mySCHelper.rawEnergy(),
                                           mySCHelper.eta(),
                                           mySCHelper.phi(),
                                           mySCHelper.r9(),
                                           mySCHelper.etaWidth(),
                                           mySCHelper.phiWidth(),
                                           mySCHelper.clustersSize(),
                                           mySCHelper.hadronicOverEm(),
                                           rho,
                                           nvertices,
                                           mySCHelper.seedEta(),
                                           mySCHelper.seedPhi(),
                                           mySCHelper.seedEnergy(),
                                           mySCHelper.e3x3(),
                                           mySCHelper.e5x5(),
                                           mySCHelper.sigmaIetaIeta(),
                                           mySCHelper.spp(),
                                           mySCHelper.sep(),
                                           mySCHelper.eMax(),
                                           mySCHelper.e2nd(),
                                           mySCHelper.eTop(),
                                           mySCHelper.eBottom(),
                                           mySCHelper.eLeft(),
                                           mySCHelper.eRight(),
                                           mySCHelper.e2x5Max(),
                                           mySCHelper.e2x5Top(),
                                           mySCHelper.e2x5Bottom(),
                                           mySCHelper.e2x5Left(),
                                           mySCHelper.e2x5Right(),
                                           mySCHelper.ietaSeed(),
                                           mySCHelper.iphiSeed(),
                                           mySCHelper.etaCrySeed(),
                                           mySCHelper.phiCrySeed(),
                                           mySCHelper.preshowerEnergyOverRaw(),
                                           ele->ecalDrivenSeed(),
                                           printDebug);
  } else if (fVersionType == kWithTrkVarV1) {
    return regressionUncertaintyWithTrkVarV1(mySCHelper.rawEnergy(),
                                             mySCHelper.eta(),
                                             mySCHelper.phi(),
                                             mySCHelper.r9(),
                                             mySCHelper.etaWidth(),
                                             mySCHelper.phiWidth(),
                                             mySCHelper.clustersSize(),
                                             mySCHelper.hadronicOverEm(),
                                             rho,
                                             nvertices,
                                             mySCHelper.seedEta(),
                                             mySCHelper.seedPhi(),
                                             mySCHelper.seedEnergy(),
                                             mySCHelper.e3x3(),
                                             mySCHelper.e5x5(),
                                             mySCHelper.sigmaIetaIeta(),
                                             mySCHelper.spp(),
                                             mySCHelper.sep(),
                                             mySCHelper.eMax(),
                                             mySCHelper.e2nd(),
                                             mySCHelper.eTop(),
                                             mySCHelper.eBottom(),
                                             mySCHelper.eLeft(),
                                             mySCHelper.eRight(),
                                             mySCHelper.e2x5Max(),
                                             mySCHelper.e2x5Top(),
                                             mySCHelper.e2x5Bottom(),
                                             mySCHelper.e2x5Left(),
                                             mySCHelper.e2x5Right(),
                                             mySCHelper.ietaSeed(),
                                             mySCHelper.iphiSeed(),
                                             mySCHelper.etaCrySeed(),
                                             mySCHelper.phiCrySeed(),
                                             mySCHelper.preshowerEnergyOverRaw(),
                                             ele->ecalDrivenSeed(),
                                             ele->trackMomentumAtVtx().R(),
                                             fmax(ele->fbrem(), -1.0),
                                             ele->charge(),
                                             fmin(ele->eSuperClusterOverP(), 20.0),
                                             ele->trackMomentumError(),
                                             ele->correctedEcalEnergyError(),
                                             ele->classification(),
                                             printDebug);
  } else if (fVersionType == kWithTrkVarV2) {
    return regressionUncertaintyWithTrkVarV2(
        mySCHelper.rawEnergy(),
        mySCHelper.eta(),
        mySCHelper.phi(),
        mySCHelper.r9(),
        mySCHelper.etaWidth(),
        mySCHelper.phiWidth(),
        mySCHelper.clustersSize(),
        mySCHelper.hadronicOverEm(),
        rho,
        nvertices,
        mySCHelper.seedEta(),
        mySCHelper.seedPhi(),
        mySCHelper.seedEnergy(),
        mySCHelper.e3x3(),
        mySCHelper.e5x5(),
        mySCHelper.sigmaIetaIeta(),
        mySCHelper.spp(),
        mySCHelper.sep(),
        mySCHelper.eMax(),
        mySCHelper.e2nd(),
        mySCHelper.eTop(),
        mySCHelper.eBottom(),
        mySCHelper.eLeft(),
        mySCHelper.eRight(),
        mySCHelper.e2x5Max(),
        mySCHelper.e2x5Top(),
        mySCHelper.e2x5Bottom(),
        mySCHelper.e2x5Left(),
        mySCHelper.e2x5Right(),
        mySCHelper.ietaSeed(),
        mySCHelper.iphiSeed(),
        mySCHelper.etaCrySeed(),
        mySCHelper.phiCrySeed(),
        mySCHelper.preshowerEnergyOverRaw(),
        ele->ecalDrivenSeed(),
        ele->trackMomentumAtVtx().R(),
        fmax(ele->fbrem(), -1.0),
        ele->charge(),
        fmin(ele->eSuperClusterOverP(), 20.0),
        ele->trackMomentumError(),
        ele->correctedEcalEnergyError(),
        ele->classification(),
        fmin(std::abs(ele->deltaEtaSuperClusterTrackAtVtx()), 0.6),
        ele->deltaPhiSuperClusterTrackAtVtx(),
        ele->deltaEtaSeedClusterTrackAtCalo(),
        ele->deltaPhiSeedClusterTrackAtCalo(),
        ele->gsfTrack()->chi2() / ele->gsfTrack()->ndof(),
        (ele->closestCtfTrackRef().isNonnull() ? ele->closestCtfTrackRef()->hitPattern().trackerLayersWithMeasurement()
                                               : -1),
        fmin(ele->eEleClusterOverPout(), 20.0),
        printDebug);
  } else {
    std::cout << "Warning: Electron Regression Type " << fVersionType
              << " is not supported. Reverting to default electron momentum.\n";
    return ele->p();
  }
}
#endif

double ElectronEnergyRegressionEvaluate::regressionValueNoTrkVar(double SCRawEnergy,
                                                                 double scEta,
                                                                 double scPhi,
                                                                 double R9,
                                                                 double etawidth,
                                                                 double phiwidth,
                                                                 double NClusters,
                                                                 double HoE,
                                                                 double rho,
                                                                 double vertices,
                                                                 double EtaSeed,
                                                                 double PhiSeed,
                                                                 double ESeed,
                                                                 double E3x3Seed,
                                                                 double E5x5Seed,
                                                                 double see,
                                                                 double spp,
                                                                 double sep,
                                                                 double EMaxSeed,
                                                                 double E2ndSeed,
                                                                 double ETopSeed,
                                                                 double EBottomSeed,
                                                                 double ELeftSeed,
                                                                 double ERightSeed,
                                                                 double E2x5MaxSeed,
                                                                 double E2x5TopSeed,
                                                                 double E2x5BottomSeed,
                                                                 double E2x5LeftSeed,
                                                                 double E2x5RightSeed,
                                                                 double IEtaSeed,
                                                                 double IPhiSeed,
                                                                 double EtaCrySeed,
                                                                 double PhiCrySeed,
                                                                 double PreShowerOverRaw,
                                                                 bool printDebug) {
  // Checking if instance has been initialized
  if (fIsInitialized == kFALSE) {
    printf("ElectronEnergyRegressionEvaluate instance not initialized !!!");
    return 0;
  }

  // Checking if type is correct
  if (!(fVersionType == kNoTrkVar)) {
    std::cout << "Error: Regression VersionType " << fVersionType
              << " is not supported to use function regressionValueNoTrkVar.\n";
    return 0;
  }
  assert(forestCorrection_ee);

  // Now applying regression according to version and (endcap/barrel)
  float *vals = (std::abs(scEta) <= 1.479) ? new float[38] : new float[31];
  if (std::abs(scEta) <= 1.479) {  // Barrel
    vals[0] = SCRawEnergy;
    vals[1] = scEta;
    vals[2] = scPhi;
    vals[3] = R9;
    vals[4] = E5x5Seed / SCRawEnergy;
    vals[5] = etawidth;
    vals[6] = phiwidth;
    vals[7] = NClusters;
    vals[8] = HoE;
    vals[9] = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed - scPhi), cos(PhiSeed - scPhi));
    vals[13] = ESeed / SCRawEnergy;
    vals[14] = E3x3Seed / ESeed;
    vals[15] = E5x5Seed / ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed / ESeed;
    vals[20] = E2ndSeed / ESeed;
    vals[21] = ETopSeed / ESeed;
    vals[22] = EBottomSeed / ESeed;
    vals[23] = ELeftSeed / ESeed;
    vals[24] = ERightSeed / ESeed;
    vals[25] = E2x5MaxSeed / ESeed;
    vals[26] = E2x5TopSeed / ESeed;
    vals[27] = E2x5BottomSeed / ESeed;
    vals[28] = E2x5LeftSeed / ESeed;
    vals[29] = E2x5RightSeed / ESeed;
    vals[30] = IEtaSeed;
    vals[31] = IPhiSeed;
    vals[32] = ((int)IEtaSeed) % 5;
    vals[33] = ((int)IPhiSeed) % 2;
    vals[34] = (std::abs(IEtaSeed) <= 25) * (((int)IEtaSeed) % 25) +
               (std::abs(IEtaSeed) > 25) * (((int)(IEtaSeed - 25 * std::abs(IEtaSeed) / IEtaSeed)) % 20);
    vals[35] = ((int)IPhiSeed) % 20;
    vals[36] = EtaCrySeed;
    vals[37] = PhiCrySeed;
  } else {  // Endcap
    vals[0] = SCRawEnergy;
    vals[1] = scEta;
    vals[2] = scPhi;
    vals[3] = R9;
    vals[4] = E5x5Seed / SCRawEnergy;
    vals[5] = etawidth;
    vals[6] = phiwidth;
    vals[7] = NClusters;
    vals[8] = HoE;
    vals[9] = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed - scPhi), cos(PhiSeed - scPhi));
    vals[13] = ESeed / SCRawEnergy;
    vals[14] = E3x3Seed / ESeed;
    vals[15] = E5x5Seed / ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed / ESeed;
    vals[20] = E2ndSeed / ESeed;
    vals[21] = ETopSeed / ESeed;
    vals[22] = EBottomSeed / ESeed;
    vals[23] = ELeftSeed / ESeed;
    vals[24] = ERightSeed / ESeed;
    vals[25] = E2x5MaxSeed / ESeed;
    vals[26] = E2x5TopSeed / ESeed;
    vals[27] = E2x5BottomSeed / ESeed;
    vals[28] = E2x5LeftSeed / ESeed;
    vals[29] = E2x5RightSeed / ESeed;
    vals[30] = PreShowerOverRaw;
  }

  // Now evaluating the regression
  double regressionResult = 0;
  Int_t BinIndex = -1;

  if (fVersionType == kNoTrkVar) {
    if (std::abs(scEta) <= 1.479) {
      regressionResult = SCRawEnergy * forestCorrection_eb->GetResponse(vals);
      BinIndex = 0;
    } else {
      regressionResult = (SCRawEnergy * (1 + PreShowerOverRaw)) * forestCorrection_ee->GetResponse(vals);
      BinIndex = 1;
    }
  }

  //print debug
  if (printDebug) {
    if (std::abs(scEta) <= 1.479) {
      std::cout << "Barrel :";
      for (unsigned int v = 0; v < 38; ++v)
        std::cout << vals[v] << ", ";
      std::cout << "\n";
    } else {
      std::cout << "Endcap :";
      for (unsigned int v = 0; v < 31; ++v)
        std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    std::cout << "BinIndex : " << BinIndex << "\n";
    std::cout << "SCRawEnergy = " << SCRawEnergy << " : PreShowerOverRaw = " << PreShowerOverRaw << std::endl;
    std::cout << "regression energy = " << regressionResult << std::endl;
  }

  // Cleaning up and returning
  delete[] vals;
  return regressionResult;
}

double ElectronEnergyRegressionEvaluate::regressionUncertaintyNoTrkVar(double SCRawEnergy,
                                                                       double scEta,
                                                                       double scPhi,
                                                                       double R9,
                                                                       double etawidth,
                                                                       double phiwidth,
                                                                       double NClusters,
                                                                       double HoE,
                                                                       double rho,
                                                                       double vertices,
                                                                       double EtaSeed,
                                                                       double PhiSeed,
                                                                       double ESeed,
                                                                       double E3x3Seed,
                                                                       double E5x5Seed,
                                                                       double see,
                                                                       double spp,
                                                                       double sep,
                                                                       double EMaxSeed,
                                                                       double E2ndSeed,
                                                                       double ETopSeed,
                                                                       double EBottomSeed,
                                                                       double ELeftSeed,
                                                                       double ERightSeed,
                                                                       double E2x5MaxSeed,
                                                                       double E2x5TopSeed,
                                                                       double E2x5BottomSeed,
                                                                       double E2x5LeftSeed,
                                                                       double E2x5RightSeed,
                                                                       double IEtaSeed,
                                                                       double IPhiSeed,
                                                                       double EtaCrySeed,
                                                                       double PhiCrySeed,
                                                                       double PreShowerOverRaw,
                                                                       bool printDebug) {
  // Checking if instance has been initialized
  if (fIsInitialized == kFALSE) {
    printf("ElectronEnergyRegressionEvaluate instance not initialized !!!");
    return 0;
  }

  // Checking if type is correct
  if (!(fVersionType == kNoTrkVar)) {
    std::cout << "Error: Regression VersionType " << fVersionType
              << " is not supported to use function regressionValueNoTrkVar.\n";
    return 0;
  }

  // Now applying regression according to version and (endcap/barrel)
  float *vals = (std::abs(scEta) <= 1.479) ? new float[38] : new float[31];
  if (std::abs(scEta) <= 1.479) {  // Barrel
    vals[0] = SCRawEnergy;
    vals[1] = scEta;
    vals[2] = scPhi;
    vals[3] = R9;
    vals[4] = E5x5Seed / SCRawEnergy;
    vals[5] = etawidth;
    vals[6] = phiwidth;
    vals[7] = NClusters;
    vals[8] = HoE;
    vals[9] = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed - scPhi), cos(PhiSeed - scPhi));
    vals[13] = ESeed / SCRawEnergy;
    vals[14] = E3x3Seed / ESeed;
    vals[15] = E5x5Seed / ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed / ESeed;
    vals[20] = E2ndSeed / ESeed;
    vals[21] = ETopSeed / ESeed;
    vals[22] = EBottomSeed / ESeed;
    vals[23] = ELeftSeed / ESeed;
    vals[24] = ERightSeed / ESeed;
    vals[25] = E2x5MaxSeed / ESeed;
    vals[26] = E2x5TopSeed / ESeed;
    vals[27] = E2x5BottomSeed / ESeed;
    vals[28] = E2x5LeftSeed / ESeed;
    vals[29] = E2x5RightSeed / ESeed;
    vals[30] = IEtaSeed;
    vals[31] = IPhiSeed;
    vals[32] = ((int)IEtaSeed) % 5;
    vals[33] = ((int)IPhiSeed) % 2;
    vals[34] = (std::abs(IEtaSeed) <= 25) * (((int)IEtaSeed) % 25) +
               (std::abs(IEtaSeed) > 25) * (((int)(IEtaSeed - 25 * std::abs(IEtaSeed) / IEtaSeed)) % 20);
    vals[35] = ((int)IPhiSeed) % 20;
    vals[36] = EtaCrySeed;
    vals[37] = PhiCrySeed;
  } else {  // Endcap
    vals[0] = SCRawEnergy;
    vals[1] = scEta;
    vals[2] = scPhi;
    vals[3] = R9;
    vals[4] = E5x5Seed / SCRawEnergy;
    vals[5] = etawidth;
    vals[6] = phiwidth;
    vals[7] = NClusters;
    vals[8] = HoE;
    vals[9] = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed - scPhi), cos(PhiSeed - scPhi));
    vals[13] = ESeed / SCRawEnergy;
    vals[14] = E3x3Seed / ESeed;
    vals[15] = E5x5Seed / ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed / ESeed;
    vals[20] = E2ndSeed / ESeed;
    vals[21] = ETopSeed / ESeed;
    vals[22] = EBottomSeed / ESeed;
    vals[23] = ELeftSeed / ESeed;
    vals[24] = ERightSeed / ESeed;
    vals[25] = E2x5MaxSeed / ESeed;
    vals[26] = E2x5TopSeed / ESeed;
    vals[27] = E2x5BottomSeed / ESeed;
    vals[28] = E2x5LeftSeed / ESeed;
    vals[29] = E2x5RightSeed / ESeed;
    vals[30] = PreShowerOverRaw;
  }

  // Now evaluating the regression
  double regressionResult = 0;
  Int_t BinIndex = -1;

  if (fVersionType == kNoTrkVar) {
    if (std::abs(scEta) <= 1.479) {
      regressionResult = SCRawEnergy * forestUncertainty_eb->GetResponse(vals);
      BinIndex = 0;
    } else {
      regressionResult = (SCRawEnergy * (1 + PreShowerOverRaw)) * forestUncertainty_ee->GetResponse(vals);
      BinIndex = 1;
    }
  }

  //print debug
  if (printDebug) {
    if (std::abs(scEta) <= 1.479) {
      std::cout << "Barrel :";
      for (unsigned int v = 0; v < 38; ++v)
        std::cout << vals[v] << ", ";
      std::cout << "\n";
    } else {
      std::cout << "Endcap :";
      for (unsigned int v = 0; v < 31; ++v)
        std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    std::cout << "BinIndex : " << BinIndex << "\n";
    std::cout << "SCRawEnergy = " << SCRawEnergy << " : PreShowerOverRaw = " << PreShowerOverRaw << std::endl;
    std::cout << "regression energy uncertainty = " << regressionResult << std::endl;
  }

  // Cleaning up and returning
  delete[] vals;
  return regressionResult;
}

double ElectronEnergyRegressionEvaluate::regressionValueNoTrkVarV1(double SCRawEnergy,
                                                                   double scEta,
                                                                   double scPhi,
                                                                   double R9,
                                                                   double etawidth,
                                                                   double phiwidth,
                                                                   double NClusters,
                                                                   double HoE,
                                                                   double rho,
                                                                   double vertices,
                                                                   double EtaSeed,
                                                                   double PhiSeed,
                                                                   double ESeed,
                                                                   double E3x3Seed,
                                                                   double E5x5Seed,
                                                                   double see,
                                                                   double spp,
                                                                   double sep,
                                                                   double EMaxSeed,
                                                                   double E2ndSeed,
                                                                   double ETopSeed,
                                                                   double EBottomSeed,
                                                                   double ELeftSeed,
                                                                   double ERightSeed,
                                                                   double E2x5MaxSeed,
                                                                   double E2x5TopSeed,
                                                                   double E2x5BottomSeed,
                                                                   double E2x5LeftSeed,
                                                                   double E2x5RightSeed,
                                                                   double IEtaSeed,
                                                                   double IPhiSeed,
                                                                   double EtaCrySeed,
                                                                   double PhiCrySeed,
                                                                   double PreShowerOverRaw,
                                                                   int IsEcalDriven,
                                                                   bool printDebug) {
  // Checking if instance has been initialized
  if (fIsInitialized == kFALSE) {
    printf("ElectronEnergyRegressionEvaluate instance not initialized !!!");
    return 0;
  }

  // Checking if type is correct
  if (!(fVersionType == kNoTrkVarV1)) {
    std::cout << "Error: Regression VersionType " << fVersionType
              << " is not supported to use function regressionValueNoTrkVar.\n";
    return 0;
  }

  // Now applying regression according to version and (endcap/barrel)
  float *vals = (std::abs(scEta) <= 1.479) ? new float[39] : new float[32];
  if (std::abs(scEta) <= 1.479) {  // Barrel
    vals[0] = SCRawEnergy;
    vals[1] = scEta;
    vals[2] = scPhi;
    vals[3] = R9;
    vals[4] = E5x5Seed / SCRawEnergy;
    vals[5] = etawidth;
    vals[6] = phiwidth;
    vals[7] = NClusters;
    vals[8] = HoE;
    vals[9] = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed - scPhi), cos(PhiSeed - scPhi));
    vals[13] = ESeed / SCRawEnergy;
    vals[14] = E3x3Seed / ESeed;
    vals[15] = E5x5Seed / ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed / ESeed;
    vals[20] = E2ndSeed / ESeed;
    vals[21] = ETopSeed / ESeed;
    vals[22] = EBottomSeed / ESeed;
    vals[23] = ELeftSeed / ESeed;
    vals[24] = ERightSeed / ESeed;
    vals[25] = E2x5MaxSeed / ESeed;
    vals[26] = E2x5TopSeed / ESeed;
    vals[27] = E2x5BottomSeed / ESeed;
    vals[28] = E2x5LeftSeed / ESeed;
    vals[29] = E2x5RightSeed / ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = IEtaSeed;
    vals[32] = IPhiSeed;
    vals[33] = ((int)IEtaSeed) % 5;
    vals[34] = ((int)IPhiSeed) % 2;
    vals[35] = (std::abs(IEtaSeed) <= 25) * (((int)IEtaSeed) % 25) +
               (std::abs(IEtaSeed) > 25) * (((int)(IEtaSeed - 25 * std::abs(IEtaSeed) / IEtaSeed)) % 20);
    vals[36] = ((int)IPhiSeed) % 20;
    vals[37] = EtaCrySeed;
    vals[38] = PhiCrySeed;
  } else {  // Endcap
    vals[0] = SCRawEnergy;
    vals[1] = scEta;
    vals[2] = scPhi;
    vals[3] = R9;
    vals[4] = E5x5Seed / SCRawEnergy;
    vals[5] = etawidth;
    vals[6] = phiwidth;
    vals[7] = NClusters;
    vals[8] = HoE;
    vals[9] = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed - scPhi), cos(PhiSeed - scPhi));
    vals[13] = ESeed / SCRawEnergy;
    vals[14] = E3x3Seed / ESeed;
    vals[15] = E5x5Seed / ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed / ESeed;
    vals[20] = E2ndSeed / ESeed;
    vals[21] = ETopSeed / ESeed;
    vals[22] = EBottomSeed / ESeed;
    vals[23] = ELeftSeed / ESeed;
    vals[24] = ERightSeed / ESeed;
    vals[25] = E2x5MaxSeed / ESeed;
    vals[26] = E2x5TopSeed / ESeed;
    vals[27] = E2x5BottomSeed / ESeed;
    vals[28] = E2x5LeftSeed / ESeed;
    vals[29] = E2x5RightSeed / ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = PreShowerOverRaw;
  }

  // Now evaluating the regression
  double regressionResult = 0;
  Int_t BinIndex = -1;

  if (fVersionType == kNoTrkVarV1) {
    if (std::abs(scEta) <= 1.479) {
      regressionResult = SCRawEnergy * forestCorrection_eb->GetResponse(vals);
      BinIndex = 0;
    } else {
      regressionResult = (SCRawEnergy * (1 + PreShowerOverRaw)) * forestCorrection_ee->GetResponse(vals);
      BinIndex = 1;
    }
  }

  //print debug
  if (printDebug) {
    if (std::abs(scEta) <= 1.479) {
      std::cout << "Barrel :";
      for (unsigned int v = 0; v < 39; ++v)
        std::cout << vals[v] << ", ";
      std::cout << "\n";
    } else {
      std::cout << "Endcap :";
      for (unsigned int v = 0; v < 32; ++v)
        std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    std::cout << "BinIndex : " << BinIndex << "\n";
    std::cout << "SCRawEnergy = " << SCRawEnergy << " : PreShowerOverRaw = " << PreShowerOverRaw << std::endl;
    std::cout << "regression energy = " << regressionResult << std::endl;
  }

  // Cleaning up and returning
  delete[] vals;
  return regressionResult;
}

double ElectronEnergyRegressionEvaluate::regressionUncertaintyNoTrkVarV1(double SCRawEnergy,
                                                                         double scEta,
                                                                         double scPhi,
                                                                         double R9,
                                                                         double etawidth,
                                                                         double phiwidth,
                                                                         double NClusters,
                                                                         double HoE,
                                                                         double rho,
                                                                         double vertices,
                                                                         double EtaSeed,
                                                                         double PhiSeed,
                                                                         double ESeed,
                                                                         double E3x3Seed,
                                                                         double E5x5Seed,
                                                                         double see,
                                                                         double spp,
                                                                         double sep,
                                                                         double EMaxSeed,
                                                                         double E2ndSeed,
                                                                         double ETopSeed,
                                                                         double EBottomSeed,
                                                                         double ELeftSeed,
                                                                         double ERightSeed,
                                                                         double E2x5MaxSeed,
                                                                         double E2x5TopSeed,
                                                                         double E2x5BottomSeed,
                                                                         double E2x5LeftSeed,
                                                                         double E2x5RightSeed,
                                                                         double IEtaSeed,
                                                                         double IPhiSeed,
                                                                         double EtaCrySeed,
                                                                         double PhiCrySeed,
                                                                         double PreShowerOverRaw,
                                                                         int IsEcalDriven,
                                                                         bool printDebug) {
  // Checking if instance has been initialized
  if (fIsInitialized == kFALSE) {
    printf("ElectronEnergyRegressionEvaluate instance not initialized !!!");
    return 0;
  }

  // Checking if type is correct
  if (!(fVersionType == kNoTrkVarV1)) {
    std::cout << "Error: Regression VersionType " << fVersionType
              << " is not supported to use function regressionValueNoTrkVar.\n";
    return 0;
  }

  // Now applying regression according to version and (endcap/barrel)
  float *vals = (std::abs(scEta) <= 1.479) ? new float[39] : new float[32];
  if (std::abs(scEta) <= 1.479) {  // Barrel
    vals[0] = SCRawEnergy;
    vals[1] = scEta;
    vals[2] = scPhi;
    vals[3] = R9;
    vals[4] = E5x5Seed / SCRawEnergy;
    vals[5] = etawidth;
    vals[6] = phiwidth;
    vals[7] = NClusters;
    vals[8] = HoE;
    vals[9] = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed - scPhi), cos(PhiSeed - scPhi));
    vals[13] = ESeed / SCRawEnergy;
    vals[14] = E3x3Seed / ESeed;
    vals[15] = E5x5Seed / ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed / ESeed;
    vals[20] = E2ndSeed / ESeed;
    vals[21] = ETopSeed / ESeed;
    vals[22] = EBottomSeed / ESeed;
    vals[23] = ELeftSeed / ESeed;
    vals[24] = ERightSeed / ESeed;
    vals[25] = E2x5MaxSeed / ESeed;
    vals[26] = E2x5TopSeed / ESeed;
    vals[27] = E2x5BottomSeed / ESeed;
    vals[28] = E2x5LeftSeed / ESeed;
    vals[29] = E2x5RightSeed / ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = IEtaSeed;
    vals[32] = IPhiSeed;
    vals[33] = ((int)IEtaSeed) % 5;
    vals[34] = ((int)IPhiSeed) % 2;
    vals[35] = (std::abs(IEtaSeed) <= 25) * (((int)IEtaSeed) % 25) +
               (std::abs(IEtaSeed) > 25) * (((int)(IEtaSeed - 25 * std::abs(IEtaSeed) / IEtaSeed)) % 20);
    vals[36] = ((int)IPhiSeed) % 20;
    vals[37] = EtaCrySeed;
    vals[38] = PhiCrySeed;
  } else {  // Endcap
    vals[0] = SCRawEnergy;
    vals[1] = scEta;
    vals[2] = scPhi;
    vals[3] = R9;
    vals[4] = E5x5Seed / SCRawEnergy;
    vals[5] = etawidth;
    vals[6] = phiwidth;
    vals[7] = NClusters;
    vals[8] = HoE;
    vals[9] = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed - scPhi), cos(PhiSeed - scPhi));
    vals[13] = ESeed / SCRawEnergy;
    vals[14] = E3x3Seed / ESeed;
    vals[15] = E5x5Seed / ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed / ESeed;
    vals[20] = E2ndSeed / ESeed;
    vals[21] = ETopSeed / ESeed;
    vals[22] = EBottomSeed / ESeed;
    vals[23] = ELeftSeed / ESeed;
    vals[24] = ERightSeed / ESeed;
    vals[25] = E2x5MaxSeed / ESeed;
    vals[26] = E2x5TopSeed / ESeed;
    vals[27] = E2x5BottomSeed / ESeed;
    vals[28] = E2x5LeftSeed / ESeed;
    vals[29] = E2x5RightSeed / ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = PreShowerOverRaw;
  }

  // Now evaluating the regression
  double regressionResult = 0;
  Int_t BinIndex = -1;

  if (fVersionType == kNoTrkVarV1) {
    if (std::abs(scEta) <= 1.479) {
      regressionResult = SCRawEnergy * forestUncertainty_eb->GetResponse(vals);
      BinIndex = 0;
    } else {
      regressionResult = (SCRawEnergy * (1 + PreShowerOverRaw)) * forestUncertainty_ee->GetResponse(vals);
      BinIndex = 1;
    }
  }

  //print debug
  if (printDebug) {
    if (std::abs(scEta) <= 1.479) {
      std::cout << "Barrel :";
      for (unsigned int v = 0; v < 39; ++v)
        std::cout << vals[v] << ", ";
      std::cout << "\n";
    } else {
      std::cout << "Endcap :";
      for (unsigned int v = 0; v < 32; ++v)
        std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    std::cout << "BinIndex : " << BinIndex << "\n";
    std::cout << "SCRawEnergy = " << SCRawEnergy << " : PreShowerOverRaw = " << PreShowerOverRaw << std::endl;
    std::cout << "regression energy uncertainty = " << regressionResult << std::endl;
  }

  // Cleaning up and returning
  delete[] vals;
  return regressionResult;
}

// This option is now deprecated. we keep it only
// for backwards compatibility
double ElectronEnergyRegressionEvaluate::regressionValueWithTrkVar(double electronP,
                                                                   double SCRawEnergy,
                                                                   double scEta,
                                                                   double scPhi,
                                                                   double R9,
                                                                   double etawidth,
                                                                   double phiwidth,
                                                                   double NClusters,
                                                                   double HoE,
                                                                   double rho,
                                                                   double vertices,
                                                                   double EtaSeed,
                                                                   double PhiSeed,
                                                                   double ESeed,
                                                                   double E3x3Seed,
                                                                   double E5x5Seed,
                                                                   double see,
                                                                   double spp,
                                                                   double sep,
                                                                   double EMaxSeed,
                                                                   double E2ndSeed,
                                                                   double ETopSeed,
                                                                   double EBottomSeed,
                                                                   double ELeftSeed,
                                                                   double ERightSeed,
                                                                   double E2x5MaxSeed,
                                                                   double E2x5TopSeed,
                                                                   double E2x5BottomSeed,
                                                                   double E2x5LeftSeed,
                                                                   double E2x5RightSeed,
                                                                   double pt,
                                                                   double GsfTrackPIn,
                                                                   double fbrem,
                                                                   double Charge,
                                                                   double EoP,
                                                                   double IEtaSeed,
                                                                   double IPhiSeed,
                                                                   double EtaCrySeed,
                                                                   double PhiCrySeed,
                                                                   double PreShowerOverRaw,
                                                                   bool printDebug) {
  // Checking if instance has been initialized
  if (fIsInitialized == kFALSE) {
    printf("ElectronEnergyRegressionEvaluate instance not initialized !!!");
    return 0;
  }

  // Checking if fVersionType is correct
  assert(fVersionType == kWithTrkVar);

  float *vals = (std::abs(scEta) <= 1.479) ? new float[43] : new float[36];
  if (std::abs(scEta) <= 1.479) {  // Barrel
    vals[0] = SCRawEnergy;
    vals[1] = scEta;
    vals[2] = scPhi;
    vals[3] = R9;
    vals[4] = E5x5Seed / SCRawEnergy;
    vals[5] = etawidth;
    vals[6] = phiwidth;
    vals[7] = NClusters;
    vals[8] = HoE;
    vals[9] = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed - scPhi), cos(PhiSeed - scPhi));
    vals[13] = ESeed / SCRawEnergy;
    vals[14] = E3x3Seed / ESeed;
    vals[15] = E5x5Seed / ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed / ESeed;
    vals[20] = E2ndSeed / ESeed;
    vals[21] = ETopSeed / ESeed;
    vals[22] = EBottomSeed / ESeed;
    vals[23] = ELeftSeed / ESeed;
    vals[24] = ERightSeed / ESeed;
    vals[25] = E2x5MaxSeed / ESeed;
    vals[26] = E2x5TopSeed / ESeed;
    vals[27] = E2x5BottomSeed / ESeed;
    vals[28] = E2x5LeftSeed / ESeed;
    vals[29] = E2x5RightSeed / ESeed;
    vals[30] = pt;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = IEtaSeed;
    vals[36] = IPhiSeed;
    vals[37] = ((int)IEtaSeed) % 5;
    vals[38] = ((int)IPhiSeed) % 2;
    vals[39] = (std::abs(IEtaSeed) <= 25) * (((int)IEtaSeed) % 25) +
               (std::abs(IEtaSeed) > 25) * (((int)(IEtaSeed - 25 * std::abs(IEtaSeed) / IEtaSeed)) % 20);
    vals[40] = ((int)IPhiSeed) % 20;
    vals[41] = EtaCrySeed;
    vals[42] = PhiCrySeed;
  }

  else {  // Endcap
    vals[0] = SCRawEnergy;
    vals[1] = scEta;
    vals[2] = scPhi;
    vals[3] = R9;
    vals[4] = E5x5Seed / SCRawEnergy;
    vals[5] = etawidth;
    vals[6] = phiwidth;
    vals[7] = NClusters;
    vals[8] = HoE;
    vals[9] = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed - scPhi), cos(PhiSeed - scPhi));
    vals[13] = ESeed / SCRawEnergy;
    vals[14] = E3x3Seed / ESeed;
    vals[15] = E5x5Seed / ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed / ESeed;
    vals[20] = E2ndSeed / ESeed;
    vals[21] = ETopSeed / ESeed;
    vals[22] = EBottomSeed / ESeed;
    vals[23] = ELeftSeed / ESeed;
    vals[24] = ERightSeed / ESeed;
    vals[25] = E2x5MaxSeed / ESeed;
    vals[26] = E2x5TopSeed / ESeed;
    vals[27] = E2x5BottomSeed / ESeed;
    vals[28] = E2x5LeftSeed / ESeed;
    vals[29] = E2x5RightSeed / ESeed;
    vals[30] = pt;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = PreShowerOverRaw;
  }

  // Now evaluating the regression
  double regressionResult = 0;

  if (fVersionType == kWithTrkVar) {
    if (std::abs(scEta) <= 1.479)
      regressionResult = SCRawEnergy * forestCorrection_eb->GetResponse(vals);
    else
      regressionResult = (SCRawEnergy * (1 + PreShowerOverRaw)) * forestCorrection_ee->GetResponse(vals);
  }

  //print debug
  if (printDebug) {
    if (scEta <= 1.479) {
      std::cout << "Barrel :";
      for (unsigned int v = 0; v < 43; ++v)
        std::cout << vals[v] << ", ";
      std::cout << "\n";
    } else {
      std::cout << "Endcap :";
      for (unsigned int v = 0; v < 36; ++v)
        std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    std::cout << "pt = " << pt << " : SCRawEnergy = " << SCRawEnergy << " : PreShowerOverRaw = " << PreShowerOverRaw
              << std::endl;
    std::cout << "regression energy = " << regressionResult << std::endl;
  }

  // Cleaning up and returning
  delete[] vals;
  return regressionResult;
}

// This option is now deprecated. we keep it only
// for backwards compatibility
double ElectronEnergyRegressionEvaluate::regressionUncertaintyWithTrkVar(double electronP,
                                                                         double SCRawEnergy,
                                                                         double scEta,
                                                                         double scPhi,
                                                                         double R9,
                                                                         double etawidth,
                                                                         double phiwidth,
                                                                         double NClusters,
                                                                         double HoE,
                                                                         double rho,
                                                                         double vertices,
                                                                         double EtaSeed,
                                                                         double PhiSeed,
                                                                         double ESeed,
                                                                         double E3x3Seed,
                                                                         double E5x5Seed,
                                                                         double see,
                                                                         double spp,
                                                                         double sep,
                                                                         double EMaxSeed,
                                                                         double E2ndSeed,
                                                                         double ETopSeed,
                                                                         double EBottomSeed,
                                                                         double ELeftSeed,
                                                                         double ERightSeed,
                                                                         double E2x5MaxSeed,
                                                                         double E2x5TopSeed,
                                                                         double E2x5BottomSeed,
                                                                         double E2x5LeftSeed,
                                                                         double E2x5RightSeed,
                                                                         double pt,
                                                                         double GsfTrackPIn,
                                                                         double fbrem,
                                                                         double Charge,
                                                                         double EoP,
                                                                         double IEtaSeed,
                                                                         double IPhiSeed,
                                                                         double EtaCrySeed,
                                                                         double PhiCrySeed,
                                                                         double PreShowerOverRaw,
                                                                         bool printDebug) {
  // Checking if instance has been initialized
  if (fIsInitialized == kFALSE) {
    printf("ElectronEnergyRegressionEvaluate instance not initialized !!!");
    return 0;
  }

  // Checking if fVersionType is correct
  assert(fVersionType == kWithTrkVar);

  float *vals = (std::abs(scEta) <= 1.479) ? new float[43] : new float[36];
  if (std::abs(scEta) <= 1.479) {  // Barrel
    vals[0] = SCRawEnergy;
    vals[1] = scEta;
    vals[2] = scPhi;
    vals[3] = R9;
    vals[4] = E5x5Seed / SCRawEnergy;
    vals[5] = etawidth;
    vals[6] = phiwidth;
    vals[7] = NClusters;
    vals[8] = HoE;
    vals[9] = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed - scPhi), cos(PhiSeed - scPhi));
    vals[13] = ESeed / SCRawEnergy;
    vals[14] = E3x3Seed / ESeed;
    vals[15] = E5x5Seed / ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed / ESeed;
    vals[20] = E2ndSeed / ESeed;
    vals[21] = ETopSeed / ESeed;
    vals[22] = EBottomSeed / ESeed;
    vals[23] = ELeftSeed / ESeed;
    vals[24] = ERightSeed / ESeed;
    vals[25] = E2x5MaxSeed / ESeed;
    vals[26] = E2x5TopSeed / ESeed;
    vals[27] = E2x5BottomSeed / ESeed;
    vals[28] = E2x5LeftSeed / ESeed;
    vals[29] = E2x5RightSeed / ESeed;
    vals[30] = pt;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = IEtaSeed;
    vals[36] = IPhiSeed;
    vals[37] = ((int)IEtaSeed) % 5;
    vals[38] = ((int)IPhiSeed) % 2;
    vals[39] = (std::abs(IEtaSeed) <= 25) * (((int)IEtaSeed) % 25) +
               (std::abs(IEtaSeed) > 25) * (((int)(IEtaSeed - 25 * std::abs(IEtaSeed) / IEtaSeed)) % 20);
    vals[40] = ((int)IPhiSeed) % 20;
    vals[41] = EtaCrySeed;
    vals[42] = PhiCrySeed;
  }

  else {  // Endcap
    vals[0] = SCRawEnergy;
    vals[1] = scEta;
    vals[2] = scPhi;
    vals[3] = R9;
    vals[4] = E5x5Seed / SCRawEnergy;
    vals[5] = etawidth;
    vals[6] = phiwidth;
    vals[7] = NClusters;
    vals[8] = HoE;
    vals[9] = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed - scPhi), cos(PhiSeed - scPhi));
    vals[13] = ESeed / SCRawEnergy;
    vals[14] = E3x3Seed / ESeed;
    vals[15] = E5x5Seed / ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed / ESeed;
    vals[20] = E2ndSeed / ESeed;
    vals[21] = ETopSeed / ESeed;
    vals[22] = EBottomSeed / ESeed;
    vals[23] = ELeftSeed / ESeed;
    vals[24] = ERightSeed / ESeed;
    vals[25] = E2x5MaxSeed / ESeed;
    vals[26] = E2x5TopSeed / ESeed;
    vals[27] = E2x5BottomSeed / ESeed;
    vals[28] = E2x5LeftSeed / ESeed;
    vals[29] = E2x5RightSeed / ESeed;
    vals[30] = pt;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = PreShowerOverRaw;
  }

  // Now evaluating the regression
  double regressionResult = 0;

  if (fVersionType == kWithTrkVar) {
    if (std::abs(scEta) <= 1.479)
      regressionResult = SCRawEnergy * forestUncertainty_eb->GetResponse(vals);
    else
      regressionResult = (SCRawEnergy * (1 + PreShowerOverRaw)) * forestUncertainty_ee->GetResponse(vals);
  }

  //print debug
  if (printDebug) {
    if (scEta <= 1.479) {
      std::cout << "Barrel :";
      for (unsigned int v = 0; v < 43; ++v)
        std::cout << vals[v] << ", ";
      std::cout << "\n";
    } else {
      std::cout << "Endcap :";
      for (unsigned int v = 0; v < 36; ++v)
        std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    std::cout << "pt = " << pt << " : SCRawEnergy = " << SCRawEnergy << " : PreShowerOverRaw = " << PreShowerOverRaw
              << std::endl;
    std::cout << "regression energy uncertainty = " << regressionResult << std::endl;
  }

  // Cleaning up and returning
  delete[] vals;
  return regressionResult;
}

double ElectronEnergyRegressionEvaluate::regressionValueWithTrkVarV1(double SCRawEnergy,
                                                                     double scEta,
                                                                     double scPhi,
                                                                     double R9,
                                                                     double etawidth,
                                                                     double phiwidth,
                                                                     double NClusters,
                                                                     double HoE,
                                                                     double rho,
                                                                     double vertices,
                                                                     double EtaSeed,
                                                                     double PhiSeed,
                                                                     double ESeed,
                                                                     double E3x3Seed,
                                                                     double E5x5Seed,
                                                                     double see,
                                                                     double spp,
                                                                     double sep,
                                                                     double EMaxSeed,
                                                                     double E2ndSeed,
                                                                     double ETopSeed,
                                                                     double EBottomSeed,
                                                                     double ELeftSeed,
                                                                     double ERightSeed,
                                                                     double E2x5MaxSeed,
                                                                     double E2x5TopSeed,
                                                                     double E2x5BottomSeed,
                                                                     double E2x5LeftSeed,
                                                                     double E2x5RightSeed,
                                                                     double IEtaSeed,
                                                                     double IPhiSeed,
                                                                     double EtaCrySeed,
                                                                     double PhiCrySeed,
                                                                     double PreShowerOverRaw,
                                                                     int IsEcalDriven,
                                                                     double GsfTrackPIn,
                                                                     double fbrem,
                                                                     double Charge,
                                                                     double EoP,
                                                                     double TrackMomentumError,
                                                                     double EcalEnergyError,
                                                                     int Classification,
                                                                     bool printDebug) {
  // Checking if instance has been initialized
  if (fIsInitialized == kFALSE) {
    printf("ElectronEnergyRegressionEvaluate instance not initialized !!!");
    return 0;
  }

  // Checking if fVersionType is correct
  assert(fVersionType == kWithTrkVarV1);

  float *vals = (std::abs(scEta) <= 1.479) ? new float[46] : new float[39];
  if (std::abs(scEta) <= 1.479) {  // Barrel
    vals[0] = SCRawEnergy;
    vals[1] = scEta;
    vals[2] = scPhi;
    vals[3] = R9;
    vals[4] = E5x5Seed / SCRawEnergy;
    vals[5] = etawidth;
    vals[6] = phiwidth;
    vals[7] = NClusters;
    vals[8] = HoE;
    vals[9] = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed - scPhi), cos(PhiSeed - scPhi));
    vals[13] = ESeed / SCRawEnergy;
    vals[14] = E3x3Seed / ESeed;
    vals[15] = E5x5Seed / ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed / ESeed;
    vals[20] = E2ndSeed / ESeed;
    vals[21] = ETopSeed / ESeed;
    vals[22] = EBottomSeed / ESeed;
    vals[23] = ELeftSeed / ESeed;
    vals[24] = ERightSeed / ESeed;
    vals[25] = E2x5MaxSeed / ESeed;
    vals[26] = E2x5TopSeed / ESeed;
    vals[27] = E2x5BottomSeed / ESeed;
    vals[28] = E2x5LeftSeed / ESeed;
    vals[29] = E2x5RightSeed / ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = TrackMomentumError / GsfTrackPIn;
    vals[36] = EcalEnergyError / SCRawEnergy;
    vals[37] = Classification;
    vals[38] = IEtaSeed;
    vals[39] = IPhiSeed;
    vals[40] = ((int)IEtaSeed) % 5;
    vals[41] = ((int)IPhiSeed) % 2;
    vals[42] = (std::abs(IEtaSeed) <= 25) * (((int)IEtaSeed) % 25) +
               (std::abs(IEtaSeed) > 25) * (((int)(IEtaSeed - 25 * std::abs(IEtaSeed) / IEtaSeed)) % 20);
    vals[43] = ((int)IPhiSeed) % 20;
    vals[44] = EtaCrySeed;
    vals[45] = PhiCrySeed;
  }

  else {  // Endcap
    vals[0] = SCRawEnergy;
    vals[1] = scEta;
    vals[2] = scPhi;
    vals[3] = R9;
    vals[4] = E5x5Seed / SCRawEnergy;
    vals[5] = etawidth;
    vals[6] = phiwidth;
    vals[7] = NClusters;
    vals[8] = HoE;
    vals[9] = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed - scPhi), cos(PhiSeed - scPhi));
    vals[13] = ESeed / SCRawEnergy;
    vals[14] = E3x3Seed / ESeed;
    vals[15] = E5x5Seed / ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed / ESeed;
    vals[20] = E2ndSeed / ESeed;
    vals[21] = ETopSeed / ESeed;
    vals[22] = EBottomSeed / ESeed;
    vals[23] = ELeftSeed / ESeed;
    vals[24] = ERightSeed / ESeed;
    vals[25] = E2x5MaxSeed / ESeed;
    vals[26] = E2x5TopSeed / ESeed;
    vals[27] = E2x5BottomSeed / ESeed;
    vals[28] = E2x5LeftSeed / ESeed;
    vals[29] = E2x5RightSeed / ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = TrackMomentumError / GsfTrackPIn;
    vals[36] = EcalEnergyError / SCRawEnergy;
    vals[37] = Classification;
    vals[38] = PreShowerOverRaw;
  }

  // Now evaluating the regression
  double regressionResult = 0;

  if (fVersionType == kWithTrkVarV1) {
    if (std::abs(scEta) <= 1.479)
      regressionResult = SCRawEnergy * forestCorrection_eb->GetResponse(vals);
    else
      regressionResult = (SCRawEnergy * (1 + PreShowerOverRaw)) * forestCorrection_ee->GetResponse(vals);
  }

  //print debug
  if (printDebug) {
    if (std::abs(scEta) <= 1.479) {
      std::cout << "Barrel :";
      for (unsigned int v = 0; v < 46; ++v)
        std::cout << vals[v] << ", ";
      std::cout << "\n";
    } else {
      std::cout << "Endcap :";
      for (unsigned int v = 0; v < 39; ++v)
        std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    std::cout << "SCRawEnergy = " << SCRawEnergy << " : PreShowerOverRaw = " << PreShowerOverRaw << std::endl;
    std::cout << "regression energy = " << regressionResult << std::endl;
  }

  // Cleaning up and returning
  delete[] vals;
  return regressionResult;
}

double ElectronEnergyRegressionEvaluate::regressionUncertaintyWithTrkVarV1(double SCRawEnergy,
                                                                           double scEta,
                                                                           double scPhi,
                                                                           double R9,
                                                                           double etawidth,
                                                                           double phiwidth,
                                                                           double NClusters,
                                                                           double HoE,
                                                                           double rho,
                                                                           double vertices,
                                                                           double EtaSeed,
                                                                           double PhiSeed,
                                                                           double ESeed,
                                                                           double E3x3Seed,
                                                                           double E5x5Seed,
                                                                           double see,
                                                                           double spp,
                                                                           double sep,
                                                                           double EMaxSeed,
                                                                           double E2ndSeed,
                                                                           double ETopSeed,
                                                                           double EBottomSeed,
                                                                           double ELeftSeed,
                                                                           double ERightSeed,
                                                                           double E2x5MaxSeed,
                                                                           double E2x5TopSeed,
                                                                           double E2x5BottomSeed,
                                                                           double E2x5LeftSeed,
                                                                           double E2x5RightSeed,
                                                                           double IEtaSeed,
                                                                           double IPhiSeed,
                                                                           double EtaCrySeed,
                                                                           double PhiCrySeed,
                                                                           double PreShowerOverRaw,
                                                                           int IsEcalDriven,
                                                                           double GsfTrackPIn,
                                                                           double fbrem,
                                                                           double Charge,
                                                                           double EoP,
                                                                           double TrackMomentumError,
                                                                           double EcalEnergyError,
                                                                           int Classification,
                                                                           bool printDebug) {
  // Checking if instance has been initialized
  if (fIsInitialized == kFALSE) {
    printf("ElectronEnergyRegressionEvaluate instance not initialized !!!");
    return 0;
  }

  // Checking if fVersionType is correct
  assert(fVersionType == kWithTrkVarV1);

  float *vals = (std::abs(scEta) <= 1.479) ? new float[46] : new float[39];
  if (std::abs(scEta) <= 1.479) {  // Barrel
    vals[0] = SCRawEnergy;
    vals[1] = scEta;
    vals[2] = scPhi;
    vals[3] = R9;
    vals[4] = E5x5Seed / SCRawEnergy;
    vals[5] = etawidth;
    vals[6] = phiwidth;
    vals[7] = NClusters;
    vals[8] = HoE;
    vals[9] = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed - scPhi), cos(PhiSeed - scPhi));
    vals[13] = ESeed / SCRawEnergy;
    vals[14] = E3x3Seed / ESeed;
    vals[15] = E5x5Seed / ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed / ESeed;
    vals[20] = E2ndSeed / ESeed;
    vals[21] = ETopSeed / ESeed;
    vals[22] = EBottomSeed / ESeed;
    vals[23] = ELeftSeed / ESeed;
    vals[24] = ERightSeed / ESeed;
    vals[25] = E2x5MaxSeed / ESeed;
    vals[26] = E2x5TopSeed / ESeed;
    vals[27] = E2x5BottomSeed / ESeed;
    vals[28] = E2x5LeftSeed / ESeed;
    vals[29] = E2x5RightSeed / ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = TrackMomentumError / GsfTrackPIn;
    vals[36] = EcalEnergyError / SCRawEnergy;
    vals[37] = Classification;
    vals[38] = IEtaSeed;
    vals[39] = IPhiSeed;
    vals[40] = ((int)IEtaSeed) % 5;
    vals[41] = ((int)IPhiSeed) % 2;
    vals[42] = (std::abs(IEtaSeed) <= 25) * (((int)IEtaSeed) % 25) +
               (std::abs(IEtaSeed) > 25) * (((int)(IEtaSeed - 25 * std::abs(IEtaSeed) / IEtaSeed)) % 20);
    vals[43] = ((int)IPhiSeed) % 20;
    vals[44] = EtaCrySeed;
    vals[45] = PhiCrySeed;
  }

  else {  // Endcap
    vals[0] = SCRawEnergy;
    vals[1] = scEta;
    vals[2] = scPhi;
    vals[3] = R9;
    vals[4] = E5x5Seed / SCRawEnergy;
    vals[5] = etawidth;
    vals[6] = phiwidth;
    vals[7] = NClusters;
    vals[8] = HoE;
    vals[9] = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed - scPhi), cos(PhiSeed - scPhi));
    vals[13] = ESeed / SCRawEnergy;
    vals[14] = E3x3Seed / ESeed;
    vals[15] = E5x5Seed / ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed / ESeed;
    vals[20] = E2ndSeed / ESeed;
    vals[21] = ETopSeed / ESeed;
    vals[22] = EBottomSeed / ESeed;
    vals[23] = ELeftSeed / ESeed;
    vals[24] = ERightSeed / ESeed;
    vals[25] = E2x5MaxSeed / ESeed;
    vals[26] = E2x5TopSeed / ESeed;
    vals[27] = E2x5BottomSeed / ESeed;
    vals[28] = E2x5LeftSeed / ESeed;
    vals[29] = E2x5RightSeed / ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = TrackMomentumError / GsfTrackPIn;
    vals[36] = EcalEnergyError / SCRawEnergy;
    vals[37] = Classification;
    vals[38] = PreShowerOverRaw;
  }

  // Now evaluating the regression
  double regressionResult = 0;

  if (fVersionType == kWithTrkVarV1) {
    if (std::abs(scEta) <= 1.479)
      regressionResult = SCRawEnergy * forestUncertainty_eb->GetResponse(vals);
    else
      regressionResult = (SCRawEnergy * (1 + PreShowerOverRaw)) * forestUncertainty_ee->GetResponse(vals);
  }

  //print debug
  if (printDebug) {
    if (std::abs(scEta) <= 1.479) {
      std::cout << "Barrel :";
      for (unsigned int v = 0; v < 46; ++v)
        std::cout << vals[v] << ", ";
      std::cout << "\n";
    } else {
      std::cout << "Endcap :";
      for (unsigned int v = 0; v < 39; ++v)
        std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    std::cout << " SCRawEnergy = " << SCRawEnergy << " : PreShowerOverRaw = " << PreShowerOverRaw << std::endl;
    std::cout << "regression energy uncertainty = " << regressionResult << std::endl;
  }

  // Cleaning up and returning
  delete[] vals;
  return regressionResult;
}

double ElectronEnergyRegressionEvaluate::regressionValueWithTrkVarV1(std::vector<double> &inputvars, bool printDebug) {
  // Checking if instance has been initialized
  if (fIsInitialized == kFALSE) {
    printf("ElectronEnergyRegressionEvaluate instance not initialized !!!");
    return 0;
  }

  // Checking if fVersionType is correct
  assert(fVersionType == kWithTrkVarV1);

  // Checking if fVersionType is correct
  assert(inputvars.size() == 42);

  double SCRawEnergy = inputvars[0];
  double scEta = inputvars[1];
  double scPhi = inputvars[2];
  double R9 = inputvars[3];
  double etawidth = inputvars[4];
  double phiwidth = inputvars[5];
  double NClusters = inputvars[6];
  double HoE = inputvars[7];
  double rho = inputvars[8];
  double vertices = inputvars[9];
  double EtaSeed = inputvars[10];
  double PhiSeed = inputvars[11];
  double ESeed = inputvars[12];
  double E3x3Seed = inputvars[13];
  double E5x5Seed = inputvars[14];
  double see = inputvars[15];
  double spp = inputvars[16];
  double sep = inputvars[17];
  double EMaxSeed = inputvars[18];
  double E2ndSeed = inputvars[19];
  double ETopSeed = inputvars[20];
  double EBottomSeed = inputvars[21];
  double ELeftSeed = inputvars[22];
  double ERightSeed = inputvars[23];
  double E2x5MaxSeed = inputvars[24];
  double E2x5TopSeed = inputvars[25];
  double E2x5BottomSeed = inputvars[26];
  double E2x5LeftSeed = inputvars[27];
  double E2x5RightSeed = inputvars[28];
  double IEtaSeed = inputvars[29];
  double IPhiSeed = inputvars[30];
  double EtaCrySeed = inputvars[31];
  double PhiCrySeed = inputvars[32];
  double PreShowerOverRaw = inputvars[33];
  int IsEcalDriven = inputvars[34];
  double GsfTrackPIn = inputvars[35];
  double fbrem = inputvars[36];
  double Charge = inputvars[37];
  double EoP = inputvars[38];
  double TrackMomentumError = inputvars[39];
  double EcalEnergyError = inputvars[40];
  int Classification = inputvars[41];

  float *vals = (std::abs(scEta) <= 1.479) ? new float[46] : new float[39];
  if (std::abs(scEta) <= 1.479) {  // Barrel
    vals[0] = SCRawEnergy;
    vals[1] = scEta;
    vals[2] = scPhi;
    vals[3] = R9;
    vals[4] = E5x5Seed / SCRawEnergy;
    vals[5] = etawidth;
    vals[6] = phiwidth;
    vals[7] = NClusters;
    vals[8] = HoE;
    vals[9] = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed - scPhi), cos(PhiSeed - scPhi));
    vals[13] = ESeed / SCRawEnergy;
    vals[14] = E3x3Seed / ESeed;
    vals[15] = E5x5Seed / ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed / ESeed;
    vals[20] = E2ndSeed / ESeed;
    vals[21] = ETopSeed / ESeed;
    vals[22] = EBottomSeed / ESeed;
    vals[23] = ELeftSeed / ESeed;
    vals[24] = ERightSeed / ESeed;
    vals[25] = E2x5MaxSeed / ESeed;
    vals[26] = E2x5TopSeed / ESeed;
    vals[27] = E2x5BottomSeed / ESeed;
    vals[28] = E2x5LeftSeed / ESeed;
    vals[29] = E2x5RightSeed / ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = TrackMomentumError / GsfTrackPIn;
    vals[36] = EcalEnergyError / SCRawEnergy;
    vals[37] = Classification;
    vals[38] = IEtaSeed;
    vals[39] = IPhiSeed;
    vals[40] = ((int)IEtaSeed) % 5;
    vals[41] = ((int)IPhiSeed) % 2;
    vals[42] = (std::abs(IEtaSeed) <= 25) * (((int)IEtaSeed) % 25) +
               (std::abs(IEtaSeed) > 25) * (((int)(IEtaSeed - 25 * std::abs(IEtaSeed) / IEtaSeed)) % 20);
    vals[43] = ((int)IPhiSeed) % 20;
    vals[44] = EtaCrySeed;
    vals[45] = PhiCrySeed;
  }

  else {  // Endcap
    vals[0] = SCRawEnergy;
    vals[1] = scEta;
    vals[2] = scPhi;
    vals[3] = R9;
    vals[4] = E5x5Seed / SCRawEnergy;
    vals[5] = etawidth;
    vals[6] = phiwidth;
    vals[7] = NClusters;
    vals[8] = HoE;
    vals[9] = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed - scPhi), cos(PhiSeed - scPhi));
    vals[13] = ESeed / SCRawEnergy;
    vals[14] = E3x3Seed / ESeed;
    vals[15] = E5x5Seed / ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed / ESeed;
    vals[20] = E2ndSeed / ESeed;
    vals[21] = ETopSeed / ESeed;
    vals[22] = EBottomSeed / ESeed;
    vals[23] = ELeftSeed / ESeed;
    vals[24] = ERightSeed / ESeed;
    vals[25] = E2x5MaxSeed / ESeed;
    vals[26] = E2x5TopSeed / ESeed;
    vals[27] = E2x5BottomSeed / ESeed;
    vals[28] = E2x5LeftSeed / ESeed;
    vals[29] = E2x5RightSeed / ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = TrackMomentumError / GsfTrackPIn;
    vals[36] = EcalEnergyError / SCRawEnergy;
    vals[37] = Classification;
    vals[38] = PreShowerOverRaw;
  }

  // Now evaluating the regression
  double regressionResult = 0;

  if (fVersionType == kWithTrkVarV1) {
    if (std::abs(scEta) <= 1.479)
      regressionResult = SCRawEnergy * forestCorrection_eb->GetResponse(vals);
    else
      regressionResult = (SCRawEnergy * (1 + PreShowerOverRaw)) * forestCorrection_ee->GetResponse(vals);
  }

  //print debug
  if (printDebug) {
    if (std::abs(scEta) <= 1.479) {
      std::cout << "Barrel :";
      for (unsigned int v = 0; v < 46; ++v)
        std::cout << vals[v] << ", ";
      std::cout << "\n";
    } else {
      std::cout << "Endcap :";
      for (unsigned int v = 0; v < 39; ++v)
        std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    std::cout << "SCRawEnergy = " << SCRawEnergy << " : PreShowerOverRaw = " << PreShowerOverRaw << std::endl;
    std::cout << "regression energy = " << regressionResult << std::endl;
  }

  // Cleaning up and returning
  delete[] vals;
  return regressionResult;
}

double ElectronEnergyRegressionEvaluate::regressionUncertaintyWithTrkVarV1(std::vector<double> &inputvars,
                                                                           bool printDebug) {
  // Checking if instance has been initialized
  if (fIsInitialized == kFALSE) {
    printf("ElectronEnergyRegressionEvaluate instance not initialized !!!");
    return 0;
  }

  // Checking if fVersionType is correct
  assert(fVersionType == kWithTrkVarV1);

  // Checking if fVersionType is correct
  assert(inputvars.size() == 42);

  double SCRawEnergy = inputvars[0];
  double scEta = inputvars[1];
  double scPhi = inputvars[2];
  double R9 = inputvars[3];
  double etawidth = inputvars[4];
  double phiwidth = inputvars[5];
  double NClusters = inputvars[6];
  double HoE = inputvars[7];
  double rho = inputvars[8];
  double vertices = inputvars[9];
  double EtaSeed = inputvars[10];
  double PhiSeed = inputvars[11];
  double ESeed = inputvars[12];
  double E3x3Seed = inputvars[13];
  double E5x5Seed = inputvars[14];
  double see = inputvars[15];
  double spp = inputvars[16];
  double sep = inputvars[17];
  double EMaxSeed = inputvars[18];
  double E2ndSeed = inputvars[19];
  double ETopSeed = inputvars[20];
  double EBottomSeed = inputvars[21];
  double ELeftSeed = inputvars[22];
  double ERightSeed = inputvars[23];
  double E2x5MaxSeed = inputvars[24];
  double E2x5TopSeed = inputvars[25];
  double E2x5BottomSeed = inputvars[26];
  double E2x5LeftSeed = inputvars[27];
  double E2x5RightSeed = inputvars[28];
  double IEtaSeed = inputvars[29];
  double IPhiSeed = inputvars[30];
  double EtaCrySeed = inputvars[31];
  double PhiCrySeed = inputvars[32];
  double PreShowerOverRaw = inputvars[33];
  int IsEcalDriven = inputvars[34];
  double GsfTrackPIn = inputvars[35];
  double fbrem = inputvars[36];
  double Charge = inputvars[37];
  double EoP = inputvars[38];
  double TrackMomentumError = inputvars[39];
  double EcalEnergyError = inputvars[40];
  int Classification = inputvars[41];

  float *vals = (std::abs(scEta) <= 1.479) ? new float[46] : new float[39];
  if (std::abs(scEta) <= 1.479) {  // Barrel
    vals[0] = SCRawEnergy;
    vals[1] = scEta;
    vals[2] = scPhi;
    vals[3] = R9;
    vals[4] = E5x5Seed / SCRawEnergy;
    vals[5] = etawidth;
    vals[6] = phiwidth;
    vals[7] = NClusters;
    vals[8] = HoE;
    vals[9] = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed - scPhi), cos(PhiSeed - scPhi));
    vals[13] = ESeed / SCRawEnergy;
    vals[14] = E3x3Seed / ESeed;
    vals[15] = E5x5Seed / ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed / ESeed;
    vals[20] = E2ndSeed / ESeed;
    vals[21] = ETopSeed / ESeed;
    vals[22] = EBottomSeed / ESeed;
    vals[23] = ELeftSeed / ESeed;
    vals[24] = ERightSeed / ESeed;
    vals[25] = E2x5MaxSeed / ESeed;
    vals[26] = E2x5TopSeed / ESeed;
    vals[27] = E2x5BottomSeed / ESeed;
    vals[28] = E2x5LeftSeed / ESeed;
    vals[29] = E2x5RightSeed / ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = TrackMomentumError / GsfTrackPIn;
    vals[36] = EcalEnergyError / SCRawEnergy;
    vals[37] = Classification;
    vals[38] = IEtaSeed;
    vals[39] = IPhiSeed;
    vals[40] = ((int)IEtaSeed) % 5;
    vals[41] = ((int)IPhiSeed) % 2;
    vals[42] = (std::abs(IEtaSeed) <= 25) * (((int)IEtaSeed) % 25) +
               (std::abs(IEtaSeed) > 25) * (((int)(IEtaSeed - 25 * std::abs(IEtaSeed) / IEtaSeed)) % 20);
    vals[43] = ((int)IPhiSeed) % 20;
    vals[44] = EtaCrySeed;
    vals[45] = PhiCrySeed;
  }

  else {  // Endcap
    vals[0] = SCRawEnergy;
    vals[1] = scEta;
    vals[2] = scPhi;
    vals[3] = R9;
    vals[4] = E5x5Seed / SCRawEnergy;
    vals[5] = etawidth;
    vals[6] = phiwidth;
    vals[7] = NClusters;
    vals[8] = HoE;
    vals[9] = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed - scPhi), cos(PhiSeed - scPhi));
    vals[13] = ESeed / SCRawEnergy;
    vals[14] = E3x3Seed / ESeed;
    vals[15] = E5x5Seed / ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed / ESeed;
    vals[20] = E2ndSeed / ESeed;
    vals[21] = ETopSeed / ESeed;
    vals[22] = EBottomSeed / ESeed;
    vals[23] = ELeftSeed / ESeed;
    vals[24] = ERightSeed / ESeed;
    vals[25] = E2x5MaxSeed / ESeed;
    vals[26] = E2x5TopSeed / ESeed;
    vals[27] = E2x5BottomSeed / ESeed;
    vals[28] = E2x5LeftSeed / ESeed;
    vals[29] = E2x5RightSeed / ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = TrackMomentumError / GsfTrackPIn;
    vals[36] = EcalEnergyError / SCRawEnergy;
    vals[37] = Classification;
    vals[38] = PreShowerOverRaw;
  }

  // Now evaluating the regression
  double regressionResult = 0;

  if (fVersionType == kWithTrkVarV1) {
    if (std::abs(scEta) <= 1.479)
      regressionResult = SCRawEnergy * forestUncertainty_eb->GetResponse(vals);
    else
      regressionResult = (SCRawEnergy * (1 + PreShowerOverRaw)) * forestUncertainty_ee->GetResponse(vals);
  }

  //print debug
  if (printDebug) {
    if (std::abs(scEta) <= 1.479) {
      std::cout << "Barrel :";
      for (unsigned int v = 0; v < 46; ++v)
        std::cout << vals[v] << ", ";
      std::cout << "\n";
    } else {
      std::cout << "Endcap :";
      for (unsigned int v = 0; v < 39; ++v)
        std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    std::cout << " SCRawEnergy = " << SCRawEnergy << " : PreShowerOverRaw = " << PreShowerOverRaw << std::endl;
    std::cout << "regression energy uncertainty = " << regressionResult << std::endl;
  }

  // Cleaning up and returning
  delete[] vals;
  return regressionResult;
}

double ElectronEnergyRegressionEvaluate::regressionValueWithTrkVarV2(double SCRawEnergy,
                                                                     double scEta,
                                                                     double scPhi,
                                                                     double R9,
                                                                     double etawidth,
                                                                     double phiwidth,
                                                                     double NClusters,
                                                                     double HoE,
                                                                     double rho,
                                                                     double vertices,
                                                                     double EtaSeed,
                                                                     double PhiSeed,
                                                                     double ESeed,
                                                                     double E3x3Seed,
                                                                     double E5x5Seed,
                                                                     double see,
                                                                     double spp,
                                                                     double sep,
                                                                     double EMaxSeed,
                                                                     double E2ndSeed,
                                                                     double ETopSeed,
                                                                     double EBottomSeed,
                                                                     double ELeftSeed,
                                                                     double ERightSeed,
                                                                     double E2x5MaxSeed,
                                                                     double E2x5TopSeed,
                                                                     double E2x5BottomSeed,
                                                                     double E2x5LeftSeed,
                                                                     double E2x5RightSeed,
                                                                     double IEtaSeed,
                                                                     double IPhiSeed,
                                                                     double EtaCrySeed,
                                                                     double PhiCrySeed,
                                                                     double PreShowerOverRaw,
                                                                     int IsEcalDriven,
                                                                     double GsfTrackPIn,
                                                                     double fbrem,
                                                                     double Charge,
                                                                     double EoP,
                                                                     double TrackMomentumError,
                                                                     double EcalEnergyError,
                                                                     int Classification,
                                                                     double detaIn,
                                                                     double dphiIn,
                                                                     double detaCalo,
                                                                     double dphiCalo,
                                                                     double GsfTrackChiSqr,
                                                                     double KFTrackNLayers,
                                                                     double ElectronEnergyOverPout,
                                                                     bool printDebug) {
  // Checking if instance has been initialized
  if (fIsInitialized == kFALSE) {
    printf("ElectronEnergyRegressionEvaluate instance not initialized !!!");
    return 0;
  }

  // Checking if fVersionType is correct
  assert(fVersionType == kWithTrkVarV2);

  float *vals = (std::abs(scEta) <= 1.479) ? new float[53] : new float[46];
  if (std::abs(scEta) <= 1.479) {  // Barrel
    vals[0] = SCRawEnergy;
    vals[1] = scEta;
    vals[2] = scPhi;
    vals[3] = R9;
    vals[4] = E5x5Seed / SCRawEnergy;
    vals[5] = etawidth;
    vals[6] = phiwidth;
    vals[7] = NClusters;
    vals[8] = HoE;
    vals[9] = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed - scPhi), cos(PhiSeed - scPhi));
    vals[13] = ESeed / SCRawEnergy;
    vals[14] = E3x3Seed / ESeed;
    vals[15] = E5x5Seed / ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed / ESeed;
    vals[20] = E2ndSeed / ESeed;
    vals[21] = ETopSeed / ESeed;
    vals[22] = EBottomSeed / ESeed;
    vals[23] = ELeftSeed / ESeed;
    vals[24] = ERightSeed / ESeed;
    vals[25] = E2x5MaxSeed / ESeed;
    vals[26] = E2x5TopSeed / ESeed;
    vals[27] = E2x5BottomSeed / ESeed;
    vals[28] = E2x5LeftSeed / ESeed;
    vals[29] = E2x5RightSeed / ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = TrackMomentumError / GsfTrackPIn;
    vals[36] = EcalEnergyError / SCRawEnergy;
    vals[37] = Classification;
    vals[38] = detaIn;
    vals[39] = dphiIn;
    vals[40] = detaCalo;
    vals[41] = dphiCalo;
    vals[42] = GsfTrackChiSqr;
    vals[43] = KFTrackNLayers;
    vals[44] = ElectronEnergyOverPout;
    vals[45] = IEtaSeed;
    vals[46] = IPhiSeed;
    vals[47] = ((int)IEtaSeed) % 5;
    vals[48] = ((int)IPhiSeed) % 2;
    vals[49] = (std::abs(IEtaSeed) <= 25) * (((int)IEtaSeed) % 25) +
               (std::abs(IEtaSeed) > 25) * (((int)(IEtaSeed - 25 * std::abs(IEtaSeed) / IEtaSeed)) % 20);
    vals[50] = ((int)IPhiSeed) % 20;
    vals[51] = EtaCrySeed;
    vals[52] = PhiCrySeed;
  }

  else {  // Endcap
    vals[0] = SCRawEnergy;
    vals[1] = scEta;
    vals[2] = scPhi;
    vals[3] = R9;
    vals[4] = E5x5Seed / SCRawEnergy;
    vals[5] = etawidth;
    vals[6] = phiwidth;
    vals[7] = NClusters;
    vals[8] = HoE;
    vals[9] = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed - scPhi), cos(PhiSeed - scPhi));
    vals[13] = ESeed / SCRawEnergy;
    vals[14] = E3x3Seed / ESeed;
    vals[15] = E5x5Seed / ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed / ESeed;
    vals[20] = E2ndSeed / ESeed;
    vals[21] = ETopSeed / ESeed;
    vals[22] = EBottomSeed / ESeed;
    vals[23] = ELeftSeed / ESeed;
    vals[24] = ERightSeed / ESeed;
    vals[25] = E2x5MaxSeed / ESeed;
    vals[26] = E2x5TopSeed / ESeed;
    vals[27] = E2x5BottomSeed / ESeed;
    vals[28] = E2x5LeftSeed / ESeed;
    vals[29] = E2x5RightSeed / ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = TrackMomentumError / GsfTrackPIn;
    vals[36] = EcalEnergyError / SCRawEnergy;
    vals[37] = Classification;
    vals[38] = detaIn;
    vals[39] = dphiIn;
    vals[40] = detaCalo;
    vals[41] = dphiCalo;
    vals[42] = GsfTrackChiSqr;
    vals[43] = KFTrackNLayers;
    vals[44] = ElectronEnergyOverPout;
    vals[45] = PreShowerOverRaw;
  }

  // Now evaluating the regression
  double regressionResult = 0;

  if (fVersionType == kWithTrkVarV2) {
    if (std::abs(scEta) <= 1.479)
      regressionResult = SCRawEnergy * forestCorrection_eb->GetResponse(vals);
    else
      regressionResult = (SCRawEnergy * (1 + PreShowerOverRaw)) * forestCorrection_ee->GetResponse(vals);
  }

  //print debug
  if (printDebug) {
    if (std::abs(scEta) <= 1.479) {
      std::cout << "Barrel :";
      for (unsigned int v = 0; v < 53; ++v)
        std::cout << vals[v] << ", ";
      std::cout << "\n";
    } else {
      std::cout << "Endcap :";
      for (unsigned int v = 0; v < 46; ++v)
        std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    std::cout << "SCRawEnergy = " << SCRawEnergy << " : PreShowerOverRaw = " << PreShowerOverRaw << std::endl;
    std::cout << "regression energy = " << regressionResult << std::endl;
  }

  // Cleaning up and returning
  delete[] vals;
  return regressionResult;
}

double ElectronEnergyRegressionEvaluate::regressionUncertaintyWithTrkVarV2(double SCRawEnergy,
                                                                           double scEta,
                                                                           double scPhi,
                                                                           double R9,
                                                                           double etawidth,
                                                                           double phiwidth,
                                                                           double NClusters,
                                                                           double HoE,
                                                                           double rho,
                                                                           double vertices,
                                                                           double EtaSeed,
                                                                           double PhiSeed,
                                                                           double ESeed,
                                                                           double E3x3Seed,
                                                                           double E5x5Seed,
                                                                           double see,
                                                                           double spp,
                                                                           double sep,
                                                                           double EMaxSeed,
                                                                           double E2ndSeed,
                                                                           double ETopSeed,
                                                                           double EBottomSeed,
                                                                           double ELeftSeed,
                                                                           double ERightSeed,
                                                                           double E2x5MaxSeed,
                                                                           double E2x5TopSeed,
                                                                           double E2x5BottomSeed,
                                                                           double E2x5LeftSeed,
                                                                           double E2x5RightSeed,
                                                                           double IEtaSeed,
                                                                           double IPhiSeed,
                                                                           double EtaCrySeed,
                                                                           double PhiCrySeed,
                                                                           double PreShowerOverRaw,
                                                                           int IsEcalDriven,
                                                                           double GsfTrackPIn,
                                                                           double fbrem,
                                                                           double Charge,
                                                                           double EoP,
                                                                           double TrackMomentumError,
                                                                           double EcalEnergyError,
                                                                           int Classification,
                                                                           double detaIn,
                                                                           double dphiIn,
                                                                           double detaCalo,
                                                                           double dphiCalo,
                                                                           double GsfTrackChiSqr,
                                                                           double KFTrackNLayers,
                                                                           double ElectronEnergyOverPout,
                                                                           bool printDebug) {
  // Checking if instance has been initialized
  if (fIsInitialized == kFALSE) {
    printf("ElectronEnergyRegressionEvaluate instance not initialized !!!");
    return 0;
  }

  // Checking if fVersionType is correct
  assert(fVersionType == kWithTrkVarV2);

  float *vals = (std::abs(scEta) <= 1.479) ? new float[53] : new float[46];
  if (std::abs(scEta) <= 1.479) {  // Barrel
    vals[0] = SCRawEnergy;
    vals[1] = scEta;
    vals[2] = scPhi;
    vals[3] = R9;
    vals[4] = E5x5Seed / SCRawEnergy;
    vals[5] = etawidth;
    vals[6] = phiwidth;
    vals[7] = NClusters;
    vals[8] = HoE;
    vals[9] = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed - scPhi), cos(PhiSeed - scPhi));
    vals[13] = ESeed / SCRawEnergy;
    vals[14] = E3x3Seed / ESeed;
    vals[15] = E5x5Seed / ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed / ESeed;
    vals[20] = E2ndSeed / ESeed;
    vals[21] = ETopSeed / ESeed;
    vals[22] = EBottomSeed / ESeed;
    vals[23] = ELeftSeed / ESeed;
    vals[24] = ERightSeed / ESeed;
    vals[25] = E2x5MaxSeed / ESeed;
    vals[26] = E2x5TopSeed / ESeed;
    vals[27] = E2x5BottomSeed / ESeed;
    vals[28] = E2x5LeftSeed / ESeed;
    vals[29] = E2x5RightSeed / ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = TrackMomentumError / GsfTrackPIn;
    vals[36] = EcalEnergyError / SCRawEnergy;
    vals[37] = Classification;
    vals[38] = detaIn;
    vals[39] = dphiIn;
    vals[40] = detaCalo;
    vals[41] = dphiCalo;
    vals[42] = GsfTrackChiSqr;
    vals[43] = KFTrackNLayers;
    vals[44] = ElectronEnergyOverPout;
    vals[45] = IEtaSeed;
    vals[46] = IPhiSeed;
    vals[47] = ((int)IEtaSeed) % 5;
    vals[48] = ((int)IPhiSeed) % 2;
    vals[49] = (std::abs(IEtaSeed) <= 25) * (((int)IEtaSeed) % 25) +
               (std::abs(IEtaSeed) > 25) * (((int)(IEtaSeed - 25 * std::abs(IEtaSeed) / IEtaSeed)) % 20);
    vals[50] = ((int)IPhiSeed) % 20;
    vals[51] = EtaCrySeed;
    vals[52] = PhiCrySeed;
  }

  else {  // Endcap
    vals[0] = SCRawEnergy;
    vals[1] = scEta;
    vals[2] = scPhi;
    vals[3] = R9;
    vals[4] = E5x5Seed / SCRawEnergy;
    vals[5] = etawidth;
    vals[6] = phiwidth;
    vals[7] = NClusters;
    vals[8] = HoE;
    vals[9] = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed - scPhi), cos(PhiSeed - scPhi));
    vals[13] = ESeed / SCRawEnergy;
    vals[14] = E3x3Seed / ESeed;
    vals[15] = E5x5Seed / ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed / ESeed;
    vals[20] = E2ndSeed / ESeed;
    vals[21] = ETopSeed / ESeed;
    vals[22] = EBottomSeed / ESeed;
    vals[23] = ELeftSeed / ESeed;
    vals[24] = ERightSeed / ESeed;
    vals[25] = E2x5MaxSeed / ESeed;
    vals[26] = E2x5TopSeed / ESeed;
    vals[27] = E2x5BottomSeed / ESeed;
    vals[28] = E2x5LeftSeed / ESeed;
    vals[29] = E2x5RightSeed / ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = TrackMomentumError / GsfTrackPIn;
    vals[36] = EcalEnergyError / SCRawEnergy;
    vals[37] = Classification;
    vals[38] = detaIn;
    vals[39] = dphiIn;
    vals[40] = detaCalo;
    vals[41] = dphiCalo;
    vals[42] = GsfTrackChiSqr;
    vals[43] = KFTrackNLayers;
    vals[44] = ElectronEnergyOverPout;
    vals[45] = PreShowerOverRaw;
  }

  // Now evaluating the regression
  double regressionResult = 0;

  if (fVersionType == kWithTrkVarV2) {
    if (std::abs(scEta) <= 1.479)
      regressionResult = SCRawEnergy * forestUncertainty_eb->GetResponse(vals);
    else
      regressionResult = (SCRawEnergy * (1 + PreShowerOverRaw)) * forestUncertainty_ee->GetResponse(vals);
  }

  //print debug
  if (printDebug) {
    if (std::abs(scEta) <= 1.479) {
      std::cout << "Barrel :";
      for (unsigned int v = 0; v < 53; ++v)
        std::cout << vals[v] << ", ";
      std::cout << "\n";
    } else {
      std::cout << "Endcap :";
      for (unsigned int v = 0; v < 46; ++v)
        std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    std::cout << "SCRawEnergy = " << SCRawEnergy << " : PreShowerOverRaw = " << PreShowerOverRaw << std::endl;
    std::cout << "regression energy uncertainty = " << regressionResult << std::endl;
  }

  // Cleaning up and returning
  delete[] vals;
  return regressionResult;
}

double ElectronEnergyRegressionEvaluate::regressionValueWithTrkVarV2(std::vector<double> &inputvars, bool printDebug) {
  // Checking if instance has been initialized
  if (fIsInitialized == kFALSE) {
    printf("ElectronEnergyRegressionEvaluate instance not initialized !!!");
    return 0;
  }

  // Checking if fVersionType is correct
  assert(fVersionType == kWithTrkVarV2);

  // Checking if fVersionType is correct
  assert(inputvars.size() == 49);

  double SCRawEnergy = inputvars[0];
  double scEta = inputvars[1];
  double scPhi = inputvars[2];
  double R9 = inputvars[3];
  double etawidth = inputvars[4];
  double phiwidth = inputvars[5];
  double NClusters = inputvars[6];
  double HoE = inputvars[7];
  double rho = inputvars[8];
  double vertices = inputvars[9];
  double EtaSeed = inputvars[10];
  double PhiSeed = inputvars[11];
  double ESeed = inputvars[12];
  double E3x3Seed = inputvars[13];
  double E5x5Seed = inputvars[14];
  double see = inputvars[15];
  double spp = inputvars[16];
  double sep = inputvars[17];
  double EMaxSeed = inputvars[18];
  double E2ndSeed = inputvars[19];
  double ETopSeed = inputvars[20];
  double EBottomSeed = inputvars[21];
  double ELeftSeed = inputvars[22];
  double ERightSeed = inputvars[23];
  double E2x5MaxSeed = inputvars[24];
  double E2x5TopSeed = inputvars[25];
  double E2x5BottomSeed = inputvars[26];
  double E2x5LeftSeed = inputvars[27];
  double E2x5RightSeed = inputvars[28];
  double IEtaSeed = inputvars[29];
  double IPhiSeed = inputvars[30];
  double EtaCrySeed = inputvars[31];
  double PhiCrySeed = inputvars[32];
  double PreShowerOverRaw = inputvars[33];
  int IsEcalDriven = inputvars[34];
  double GsfTrackPIn = inputvars[35];
  double fbrem = inputvars[36];
  double Charge = inputvars[37];
  double EoP = inputvars[38];
  double TrackMomentumError = inputvars[39];
  double EcalEnergyError = inputvars[40];
  int Classification = inputvars[41];
  double detaIn = inputvars[42];
  double dphiIn = inputvars[43];
  double detaCalo = inputvars[44];
  double dphiCalo = inputvars[45];
  double GsfTrackChiSqr = inputvars[46];
  double KFTrackNLayers = inputvars[47];
  double ElectronEnergyOverPout = inputvars[48];

  float *vals = (std::abs(scEta) <= 1.479) ? new float[53] : new float[46];
  if (std::abs(scEta) <= 1.479) {  // Barrel
    vals[0] = SCRawEnergy;
    vals[1] = scEta;
    vals[2] = scPhi;
    vals[3] = R9;
    vals[4] = E5x5Seed / SCRawEnergy;
    vals[5] = etawidth;
    vals[6] = phiwidth;
    vals[7] = NClusters;
    vals[8] = HoE;
    vals[9] = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed - scPhi), cos(PhiSeed - scPhi));
    vals[13] = ESeed / SCRawEnergy;
    vals[14] = E3x3Seed / ESeed;
    vals[15] = E5x5Seed / ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed / ESeed;
    vals[20] = E2ndSeed / ESeed;
    vals[21] = ETopSeed / ESeed;
    vals[22] = EBottomSeed / ESeed;
    vals[23] = ELeftSeed / ESeed;
    vals[24] = ERightSeed / ESeed;
    vals[25] = E2x5MaxSeed / ESeed;
    vals[26] = E2x5TopSeed / ESeed;
    vals[27] = E2x5BottomSeed / ESeed;
    vals[28] = E2x5LeftSeed / ESeed;
    vals[29] = E2x5RightSeed / ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = TrackMomentumError / GsfTrackPIn;
    vals[36] = EcalEnergyError / SCRawEnergy;
    vals[37] = Classification;
    vals[38] = detaIn;
    vals[39] = dphiIn;
    vals[40] = detaCalo;
    vals[41] = dphiCalo;
    vals[42] = GsfTrackChiSqr;
    vals[43] = KFTrackNLayers;
    vals[44] = ElectronEnergyOverPout;
    vals[45] = IEtaSeed;
    vals[46] = IPhiSeed;
    vals[47] = ((int)IEtaSeed) % 5;
    vals[48] = ((int)IPhiSeed) % 2;
    vals[49] = (std::abs(IEtaSeed) <= 25) * (((int)IEtaSeed) % 25) +
               (std::abs(IEtaSeed) > 25) * (((int)(IEtaSeed - 25 * std::abs(IEtaSeed) / IEtaSeed)) % 20);
    vals[50] = ((int)IPhiSeed) % 20;
    vals[51] = EtaCrySeed;
    vals[52] = PhiCrySeed;
  }

  else {  // Endcap
    vals[0] = SCRawEnergy;
    vals[1] = scEta;
    vals[2] = scPhi;
    vals[3] = R9;
    vals[4] = E5x5Seed / SCRawEnergy;
    vals[5] = etawidth;
    vals[6] = phiwidth;
    vals[7] = NClusters;
    vals[8] = HoE;
    vals[9] = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed - scPhi), cos(PhiSeed - scPhi));
    vals[13] = ESeed / SCRawEnergy;
    vals[14] = E3x3Seed / ESeed;
    vals[15] = E5x5Seed / ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed / ESeed;
    vals[20] = E2ndSeed / ESeed;
    vals[21] = ETopSeed / ESeed;
    vals[22] = EBottomSeed / ESeed;
    vals[23] = ELeftSeed / ESeed;
    vals[24] = ERightSeed / ESeed;
    vals[25] = E2x5MaxSeed / ESeed;
    vals[26] = E2x5TopSeed / ESeed;
    vals[27] = E2x5BottomSeed / ESeed;
    vals[28] = E2x5LeftSeed / ESeed;
    vals[29] = E2x5RightSeed / ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = TrackMomentumError / GsfTrackPIn;
    vals[36] = EcalEnergyError / SCRawEnergy;
    vals[37] = Classification;
    vals[38] = detaIn;
    vals[39] = dphiIn;
    vals[40] = detaCalo;
    vals[41] = dphiCalo;
    vals[42] = GsfTrackChiSqr;
    vals[43] = KFTrackNLayers;
    vals[44] = ElectronEnergyOverPout;
    vals[45] = PreShowerOverRaw;
  }

  // Now evaluating the regression
  double regressionResult = 0;

  if (fVersionType == kWithTrkVarV2) {
    if (std::abs(scEta) <= 1.479)
      regressionResult = SCRawEnergy * forestCorrection_eb->GetResponse(vals);
    else
      regressionResult = (SCRawEnergy * (1 + PreShowerOverRaw)) * forestCorrection_ee->GetResponse(vals);
  }

  //print debug
  if (printDebug) {
    if (std::abs(scEta) <= 1.479) {
      std::cout << "Barrel :";
      for (unsigned int v = 0; v < 53; ++v)
        std::cout << vals[v] << ", ";
      std::cout << "\n";
    } else {
      std::cout << "Endcap :";
      for (unsigned int v = 0; v < 46; ++v)
        std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    std::cout << "SCRawEnergy = " << SCRawEnergy << " : PreShowerOverRaw = " << PreShowerOverRaw << std::endl;
    std::cout << "regression energy = " << regressionResult << std::endl;
  }

  // Cleaning up and returning
  delete[] vals;
  return regressionResult;
}

double ElectronEnergyRegressionEvaluate::regressionUncertaintyWithTrkVarV2(std::vector<double> &inputvars,
                                                                           bool printDebug) {
  // Checking if instance has been initialized
  if (fIsInitialized == kFALSE) {
    printf("ElectronEnergyRegressionEvaluate instance not initialized !!!");
    return 0;
  }

  // Checking if fVersionType is correct
  assert(fVersionType == kWithTrkVarV2);

  // Checking if fVersionType is correct
  assert(inputvars.size() == 49);

  double SCRawEnergy = inputvars[0];
  double scEta = inputvars[1];
  double scPhi = inputvars[2];
  double R9 = inputvars[3];
  double etawidth = inputvars[4];
  double phiwidth = inputvars[5];
  double NClusters = inputvars[6];
  double HoE = inputvars[7];
  double rho = inputvars[8];
  double vertices = inputvars[9];
  double EtaSeed = inputvars[10];
  double PhiSeed = inputvars[11];
  double ESeed = inputvars[12];
  double E3x3Seed = inputvars[13];
  double E5x5Seed = inputvars[14];
  double see = inputvars[15];
  double spp = inputvars[16];
  double sep = inputvars[17];
  double EMaxSeed = inputvars[18];
  double E2ndSeed = inputvars[19];
  double ETopSeed = inputvars[20];
  double EBottomSeed = inputvars[21];
  double ELeftSeed = inputvars[22];
  double ERightSeed = inputvars[23];
  double E2x5MaxSeed = inputvars[24];
  double E2x5TopSeed = inputvars[25];
  double E2x5BottomSeed = inputvars[26];
  double E2x5LeftSeed = inputvars[27];
  double E2x5RightSeed = inputvars[28];
  double IEtaSeed = inputvars[29];
  double IPhiSeed = inputvars[30];
  double EtaCrySeed = inputvars[31];
  double PhiCrySeed = inputvars[32];
  double PreShowerOverRaw = inputvars[33];
  int IsEcalDriven = inputvars[34];
  double GsfTrackPIn = inputvars[35];
  double fbrem = inputvars[36];
  double Charge = inputvars[37];
  double EoP = inputvars[38];
  double TrackMomentumError = inputvars[39];
  double EcalEnergyError = inputvars[40];
  int Classification = inputvars[41];
  double detaIn = inputvars[42];
  double dphiIn = inputvars[43];
  double detaCalo = inputvars[44];
  double dphiCalo = inputvars[45];
  double GsfTrackChiSqr = inputvars[46];
  double KFTrackNLayers = inputvars[47];
  double ElectronEnergyOverPout = inputvars[48];

  float *vals = (std::abs(scEta) <= 1.479) ? new float[53] : new float[46];
  if (std::abs(scEta) <= 1.479) {  // Barrel
    vals[0] = SCRawEnergy;
    vals[1] = scEta;
    vals[2] = scPhi;
    vals[3] = R9;
    vals[4] = E5x5Seed / SCRawEnergy;
    vals[5] = etawidth;
    vals[6] = phiwidth;
    vals[7] = NClusters;
    vals[8] = HoE;
    vals[9] = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed - scPhi), cos(PhiSeed - scPhi));
    vals[13] = ESeed / SCRawEnergy;
    vals[14] = E3x3Seed / ESeed;
    vals[15] = E5x5Seed / ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed / ESeed;
    vals[20] = E2ndSeed / ESeed;
    vals[21] = ETopSeed / ESeed;
    vals[22] = EBottomSeed / ESeed;
    vals[23] = ELeftSeed / ESeed;
    vals[24] = ERightSeed / ESeed;
    vals[25] = E2x5MaxSeed / ESeed;
    vals[26] = E2x5TopSeed / ESeed;
    vals[27] = E2x5BottomSeed / ESeed;
    vals[28] = E2x5LeftSeed / ESeed;
    vals[29] = E2x5RightSeed / ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = TrackMomentumError / GsfTrackPIn;
    vals[36] = EcalEnergyError / SCRawEnergy;
    vals[37] = Classification;
    vals[38] = detaIn;
    vals[39] = dphiIn;
    vals[40] = detaCalo;
    vals[41] = dphiCalo;
    vals[42] = GsfTrackChiSqr;
    vals[43] = KFTrackNLayers;
    vals[44] = ElectronEnergyOverPout;
    vals[45] = IEtaSeed;
    vals[46] = IPhiSeed;
    vals[47] = ((int)IEtaSeed) % 5;
    vals[48] = ((int)IPhiSeed) % 2;
    vals[49] = (std::abs(IEtaSeed) <= 25) * (((int)IEtaSeed) % 25) +
               (std::abs(IEtaSeed) > 25) * (((int)(IEtaSeed - 25 * std::abs(IEtaSeed) / IEtaSeed)) % 20);
    vals[50] = ((int)IPhiSeed) % 20;
    vals[51] = EtaCrySeed;
    vals[52] = PhiCrySeed;
  }

  else {  // Endcap
    vals[0] = SCRawEnergy;
    vals[1] = scEta;
    vals[2] = scPhi;
    vals[3] = R9;
    vals[4] = E5x5Seed / SCRawEnergy;
    vals[5] = etawidth;
    vals[6] = phiwidth;
    vals[7] = NClusters;
    vals[8] = HoE;
    vals[9] = rho;
    vals[10] = vertices;
    vals[11] = EtaSeed - scEta;
    vals[12] = atan2(sin(PhiSeed - scPhi), cos(PhiSeed - scPhi));
    vals[13] = ESeed / SCRawEnergy;
    vals[14] = E3x3Seed / ESeed;
    vals[15] = E5x5Seed / ESeed;
    vals[16] = see;
    vals[17] = spp;
    vals[18] = sep;
    vals[19] = EMaxSeed / ESeed;
    vals[20] = E2ndSeed / ESeed;
    vals[21] = ETopSeed / ESeed;
    vals[22] = EBottomSeed / ESeed;
    vals[23] = ELeftSeed / ESeed;
    vals[24] = ERightSeed / ESeed;
    vals[25] = E2x5MaxSeed / ESeed;
    vals[26] = E2x5TopSeed / ESeed;
    vals[27] = E2x5BottomSeed / ESeed;
    vals[28] = E2x5LeftSeed / ESeed;
    vals[29] = E2x5RightSeed / ESeed;
    vals[30] = IsEcalDriven;
    vals[31] = GsfTrackPIn;
    vals[32] = fbrem;
    vals[33] = Charge;
    vals[34] = EoP;
    vals[35] = TrackMomentumError / GsfTrackPIn;
    vals[36] = EcalEnergyError / SCRawEnergy;
    vals[37] = Classification;
    vals[38] = detaIn;
    vals[39] = dphiIn;
    vals[40] = detaCalo;
    vals[41] = dphiCalo;
    vals[42] = GsfTrackChiSqr;
    vals[43] = KFTrackNLayers;
    vals[44] = ElectronEnergyOverPout;
    vals[45] = PreShowerOverRaw;
  }

  // Now evaluating the regression
  double regressionResult = 0;

  if (fVersionType == kWithTrkVarV2) {
    if (std::abs(scEta) <= 1.479)
      regressionResult = SCRawEnergy * forestUncertainty_eb->GetResponse(vals);
    else
      regressionResult = (SCRawEnergy * (1 + PreShowerOverRaw)) * forestUncertainty_ee->GetResponse(vals);
  }

  //print debug
  if (printDebug) {
    if (std::abs(scEta) <= 1.479) {
      std::cout << "Barrel :";
      for (unsigned int v = 0; v < 53; ++v)
        std::cout << vals[v] << ", ";
      std::cout << "\n";
    } else {
      std::cout << "Endcap :";
      for (unsigned int v = 0; v < 46; ++v)
        std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    std::cout << "SCRawEnergy = " << SCRawEnergy << " : PreShowerOverRaw = " << PreShowerOverRaw << std::endl;
    std::cout << "regression energy uncertainty = " << regressionResult << std::endl;
  }

  // Cleaning up and returning
  delete[] vals;
  return regressionResult;
}

double ElectronEnergyRegressionEvaluate::regressionValueWithSubClusters(double SCRawEnergy,
                                                                        double scEta,
                                                                        double scPhi,
                                                                        double R9,
                                                                        double etawidth,
                                                                        double phiwidth,
                                                                        double NClusters,
                                                                        double HoE,
                                                                        double rho,
                                                                        double vertices,
                                                                        double EtaSeed,
                                                                        double PhiSeed,
                                                                        double ESeed,
                                                                        double E3x3Seed,
                                                                        double E5x5Seed,
                                                                        double see,
                                                                        double spp,
                                                                        double sep,
                                                                        double EMaxSeed,
                                                                        double E2ndSeed,
                                                                        double ETopSeed,
                                                                        double EBottomSeed,
                                                                        double ELeftSeed,
                                                                        double ERightSeed,
                                                                        double E2x5MaxSeed,
                                                                        double E2x5TopSeed,
                                                                        double E2x5BottomSeed,
                                                                        double E2x5LeftSeed,
                                                                        double E2x5RightSeed,
                                                                        double IEtaSeed,
                                                                        double IPhiSeed,
                                                                        double EtaCrySeed,
                                                                        double PhiCrySeed,
                                                                        double PreShowerOverRaw,
                                                                        double isEcalDriven,
                                                                        double isEtaGap,
                                                                        double isPhiGap,
                                                                        double isDeeGap,
                                                                        double ESubs,
                                                                        double ESub1,
                                                                        double EtaSub1,
                                                                        double PhiSub1,
                                                                        double EMaxSub1,
                                                                        double E3x3Sub1,
                                                                        double ESub2,
                                                                        double EtaSub2,
                                                                        double PhiSub2,
                                                                        double EMaxSub2,
                                                                        double E3x3Sub2,
                                                                        double ESub3,
                                                                        double EtaSub3,
                                                                        double PhiSub3,
                                                                        double EMaxSub3,
                                                                        double E3x3Sub3,
                                                                        double NPshwClusters,
                                                                        double EPshwSubs,
                                                                        double EPshwSub1,
                                                                        double EtaPshwSub1,
                                                                        double PhiPshwSub1,
                                                                        double EPshwSub2,
                                                                        double EtaPshwSub2,
                                                                        double PhiPshwSub2,
                                                                        double EPshwSub3,
                                                                        double EtaPshwSub3,
                                                                        double PhiPshwSub3,
                                                                        bool isEB,
                                                                        bool printDebug) {
  // Checking if instance has been initialized
  if (fIsInitialized == kFALSE) {
    printf("ElectronEnergyRegressionEvaluate instance not initialized !!!");
    return 0;
  }

  // Checking if type is correct
  if (!(fVersionType == kWithSubCluVar)) {
    std::cout << "Error: Regression VersionType " << fVersionType
              << " is not supported to use function regressionValueWithSubClusters.\n";
    return 0;
  }

  // Now applying regression according to version and (endcap/barrel)
  float *vals = (isEB) ? new float[61] : new float[65];
  if (isEB) {  // Barrel
    vals[0] = rho;
    vals[1] = vertices;
    vals[2] = isEcalDriven;
    vals[3] = isEtaGap;
    vals[4] = isPhiGap;
    vals[5] = isDeeGap;
    vals[6] = SCRawEnergy;
    vals[7] = scEta;
    vals[8] = scPhi;
    vals[9] = R9;
    vals[10] = etawidth;
    vals[11] = phiwidth;
    vals[12] = NClusters;
    vals[13] = HoE;
    vals[14] = EtaSeed - scEta;
    vals[15] = atan2(sin(PhiSeed - scPhi), cos(PhiSeed - scPhi));
    vals[16] = ESeed / SCRawEnergy;
    vals[17] = E3x3Seed / ESeed;
    vals[18] = E5x5Seed / SCRawEnergy;
    vals[19] = E5x5Seed / ESeed;
    vals[20] = EMaxSeed / ESeed;
    vals[21] = E2ndSeed / ESeed;
    vals[22] = ETopSeed / ESeed;
    vals[23] = EBottomSeed / ESeed;
    vals[24] = ELeftSeed / ESeed;
    vals[25] = ERightSeed / ESeed;
    vals[26] = E2x5MaxSeed / ESeed;
    vals[27] = E2x5TopSeed / ESeed;
    vals[28] = E2x5BottomSeed / ESeed;
    vals[29] = E2x5LeftSeed / ESeed;
    vals[30] = E2x5RightSeed / ESeed;
    vals[31] = see;
    vals[32] = spp;
    vals[33] = sep;
    vals[34] = phiwidth / etawidth;
    vals[35] = (ELeftSeed + ERightSeed == 0. ? 0. : (ELeftSeed - ERightSeed) / (ELeftSeed + ERightSeed));
    vals[36] = (ETopSeed + EBottomSeed == 0. ? 0. : (ETopSeed - EBottomSeed) / (ETopSeed + EBottomSeed));
    vals[37] = ESubs / SCRawEnergy;
    vals[38] = ESub1 / SCRawEnergy;
    vals[39] = (NClusters <= 1 ? 999. : EtaSub1 - EtaSeed);
    vals[40] = (NClusters <= 1 ? 999. : atan2(sin(PhiSub1 - PhiSeed), cos(PhiSub1 - PhiSeed)));
    vals[41] = (NClusters <= 1 ? 0. : EMaxSub1 / ESub1);
    vals[42] = (NClusters <= 1 ? 0. : E3x3Sub1 / ESub1);
    vals[43] = ESub2 / SCRawEnergy;
    vals[44] = (NClusters <= 2 ? 999. : EtaSub2 - EtaSeed);
    vals[45] = (NClusters <= 2 ? 999. : atan2(sin(PhiSub2 - PhiSeed), cos(PhiSub2 - PhiSeed)));
    vals[46] = (NClusters <= 2 ? 0. : EMaxSub2 / ESub2);
    vals[47] = (NClusters <= 2 ? 0. : E3x3Sub2 / ESub2);
    vals[48] = ESub3 / SCRawEnergy;
    vals[49] = (NClusters <= 3 ? 999. : EtaSub3 - EtaSeed);
    vals[50] = (NClusters <= 3 ? 999. : atan2(sin(PhiSub3 - PhiSeed), cos(PhiSub3 - PhiSeed)));
    vals[51] = (NClusters <= 3 ? 0. : EMaxSub3 / ESub3);
    vals[52] = (NClusters <= 3 ? 0. : E3x3Sub3 / ESub3);
    vals[53] = IEtaSeed;
    vals[54] = ((int)IEtaSeed) % 5;
    vals[55] = (std::abs(IEtaSeed) <= 25) * (((int)IEtaSeed) % 25) +
               (std::abs(IEtaSeed) > 25) * (((int)(IEtaSeed - 25 * std::abs(IEtaSeed) / IEtaSeed)) % 20);
    vals[56] = IPhiSeed;
    vals[57] = ((int)IPhiSeed) % 2;
    vals[58] = ((int)IPhiSeed) % 20;
    vals[59] = EtaCrySeed;
    vals[60] = PhiCrySeed;
  } else {  // Endcap
    vals[0] = rho;
    vals[1] = vertices;
    vals[2] = isEcalDriven;
    vals[3] = isEtaGap;
    vals[4] = isPhiGap;
    vals[5] = isDeeGap;
    vals[6] = SCRawEnergy;
    vals[7] = scEta;
    vals[8] = scPhi;
    vals[9] = R9;
    vals[10] = etawidth;
    vals[11] = phiwidth;
    vals[12] = NClusters;
    vals[13] = HoE;
    vals[14] = EtaSeed - scEta;
    vals[15] = atan2(sin(PhiSeed - scPhi), cos(PhiSeed - scPhi));
    vals[16] = ESeed / SCRawEnergy;
    vals[17] = E3x3Seed / ESeed;
    vals[18] = E5x5Seed / SCRawEnergy;
    vals[19] = E5x5Seed / ESeed;
    vals[20] = EMaxSeed / ESeed;
    vals[21] = E2ndSeed / ESeed;
    vals[22] = ETopSeed / ESeed;
    vals[23] = EBottomSeed / ESeed;
    vals[24] = ELeftSeed / ESeed;
    vals[25] = ERightSeed / ESeed;
    vals[26] = E2x5MaxSeed / ESeed;
    vals[27] = E2x5TopSeed / ESeed;
    vals[28] = E2x5BottomSeed / ESeed;
    vals[29] = E2x5LeftSeed / ESeed;
    vals[30] = E2x5RightSeed / ESeed;
    vals[31] = see;
    vals[32] = spp;
    vals[33] = sep;
    vals[34] = phiwidth / etawidth;
    vals[35] = (ELeftSeed + ERightSeed == 0. ? 0. : (ELeftSeed - ERightSeed) / (ELeftSeed + ERightSeed));
    vals[36] = (ETopSeed + EBottomSeed == 0. ? 0. : (ETopSeed - EBottomSeed) / (ETopSeed + EBottomSeed));
    vals[37] = ESubs / SCRawEnergy;
    vals[38] = ESub1 / SCRawEnergy;
    vals[39] = (NClusters <= 1 ? 999. : EtaSub1 - EtaSeed);
    vals[40] = (NClusters <= 1 ? 999. : atan2(sin(PhiSub1 - PhiSeed), cos(PhiSub1 - PhiSeed)));
    vals[41] = (NClusters <= 1 ? 0. : EMaxSub1 / ESub1);
    vals[42] = (NClusters <= 1 ? 0. : E3x3Sub1 / ESub1);
    vals[43] = ESub2 / SCRawEnergy;
    vals[44] = (NClusters <= 2 ? 999. : EtaSub2 - EtaSeed);
    vals[45] = (NClusters <= 2 ? 999. : atan2(sin(PhiSub2 - PhiSeed), cos(PhiSub2 - PhiSeed)));
    vals[46] = (NClusters <= 2 ? 0. : EMaxSub2 / ESub2);
    vals[47] = (NClusters <= 2 ? 0. : E3x3Sub2 / ESub2);
    vals[48] = ESub3 / SCRawEnergy;
    vals[49] = (NClusters <= 3 ? 999. : EtaSub3 - EtaSeed);
    vals[50] = (NClusters <= 3 ? 999. : atan2(sin(PhiSub3 - PhiSeed), cos(PhiSub3 - PhiSeed)));
    vals[51] = (NClusters <= 3 ? 0. : EMaxSub3 / ESub3);
    vals[52] = (NClusters <= 3 ? 0. : E3x3Sub3 / ESub3);
    vals[53] = PreShowerOverRaw;
    vals[54] = NPshwClusters;
    vals[55] = EPshwSubs / SCRawEnergy;
    vals[56] = EPshwSub1 / SCRawEnergy;
    vals[57] = (NPshwClusters == 0 ? 999. : EtaPshwSub1 - EtaSeed);
    vals[58] = (NPshwClusters == 0 ? 999. : atan2(sin(PhiPshwSub1 - PhiSeed), cos(PhiPshwSub1 - PhiSeed)));
    vals[59] = EPshwSub2 / SCRawEnergy;
    vals[60] = (NPshwClusters <= 1 ? 999. : EtaPshwSub2 - EtaSeed);
    vals[61] = (NPshwClusters <= 1 ? 999. : atan2(sin(PhiPshwSub2 - PhiSeed), cos(PhiPshwSub2 - PhiSeed)));
    vals[62] = EPshwSub3 / SCRawEnergy;
    vals[63] = (NPshwClusters <= 2 ? 999. : EtaPshwSub3 - EtaSeed);
    vals[64] = (NPshwClusters <= 2 ? 999. : atan2(sin(PhiPshwSub3 - PhiSeed), cos(PhiPshwSub3 - PhiSeed)));
  }

  // Now evaluating the regression
  double regressionResult = 0;
  Int_t BinIndex = -1;

  if (fVersionType == kWithSubCluVar) {
    if (isEB) {
      regressionResult = SCRawEnergy * forestCorrection_eb->GetResponse(vals);
      BinIndex = 0;
    } else {
      regressionResult = (SCRawEnergy * (1 + PreShowerOverRaw)) * forestCorrection_ee->GetResponse(vals);
      BinIndex = 1;
    }
  }

  //print debug
  if (printDebug) {
    if (isEB) {
      std::cout << "Barrel :";
      for (unsigned int v = 0; v < 61; ++v)
        std::cout << v << "=" << vals[v] << ", ";
      std::cout << "\n";
    } else {
      std::cout << "Endcap :";
      for (unsigned int v = 0; v < 65; ++v)
        std::cout << v << "=" << vals[v] << ", ";
      std::cout << "\n";
    }
    std::cout << "BinIndex : " << BinIndex << "\n";
    std::cout << "SCRawEnergy = " << SCRawEnergy << " : PreShowerOverRaw = " << PreShowerOverRaw << std::endl;
    std::cout << "regression energy = " << regressionResult << std::endl;
  }

  // Cleaning up and returning
  delete[] vals;
  return regressionResult;
}

double ElectronEnergyRegressionEvaluate::regressionUncertaintyWithSubClusters(double SCRawEnergy,
                                                                              double scEta,
                                                                              double scPhi,
                                                                              double R9,
                                                                              double etawidth,
                                                                              double phiwidth,
                                                                              double NClusters,
                                                                              double HoE,
                                                                              double rho,
                                                                              double vertices,
                                                                              double EtaSeed,
                                                                              double PhiSeed,
                                                                              double ESeed,
                                                                              double E3x3Seed,
                                                                              double E5x5Seed,
                                                                              double see,
                                                                              double spp,
                                                                              double sep,
                                                                              double EMaxSeed,
                                                                              double E2ndSeed,
                                                                              double ETopSeed,
                                                                              double EBottomSeed,
                                                                              double ELeftSeed,
                                                                              double ERightSeed,
                                                                              double E2x5MaxSeed,
                                                                              double E2x5TopSeed,
                                                                              double E2x5BottomSeed,
                                                                              double E2x5LeftSeed,
                                                                              double E2x5RightSeed,
                                                                              double IEtaSeed,
                                                                              double IPhiSeed,
                                                                              double EtaCrySeed,
                                                                              double PhiCrySeed,
                                                                              double PreShowerOverRaw,
                                                                              double isEcalDriven,
                                                                              double isEtaGap,
                                                                              double isPhiGap,
                                                                              double isDeeGap,
                                                                              double ESubs,
                                                                              double ESub1,
                                                                              double EtaSub1,
                                                                              double PhiSub1,
                                                                              double EMaxSub1,
                                                                              double E3x3Sub1,
                                                                              double ESub2,
                                                                              double EtaSub2,
                                                                              double PhiSub2,
                                                                              double EMaxSub2,
                                                                              double E3x3Sub2,
                                                                              double ESub3,
                                                                              double EtaSub3,
                                                                              double PhiSub3,
                                                                              double EMaxSub3,
                                                                              double E3x3Sub3,
                                                                              double NPshwClusters,
                                                                              double EPshwSubs,
                                                                              double EPshwSub1,
                                                                              double EtaPshwSub1,
                                                                              double PhiPshwSub1,
                                                                              double EPshwSub2,
                                                                              double EtaPshwSub2,
                                                                              double PhiPshwSub2,
                                                                              double EPshwSub3,
                                                                              double EtaPshwSub3,
                                                                              double PhiPshwSub3,
                                                                              bool isEB,
                                                                              bool printDebug) {
  // Checking if instance has been initialized
  if (fIsInitialized == kFALSE) {
    printf("ElectronEnergyRegressionEvaluate instance not initialized !!!");
    return 0;
  }

  // Checking if type is correct
  if (!(fVersionType == kWithSubCluVar)) {
    std::cout << "Error: Regression VersionType " << fVersionType
              << " is not supported to use function regressionValueWithSubClusters.\n";
    return 0;
  }

  // Now applying regression according to version and (endcap/barrel)
  float *vals = (isEB) ? new float[61] : new float[65];
  if (isEB) {  // Barrel
    vals[0] = rho;
    vals[1] = vertices;
    vals[2] = isEcalDriven;
    vals[3] = isEtaGap;
    vals[4] = isPhiGap;
    vals[5] = isDeeGap;
    vals[6] = SCRawEnergy;
    vals[7] = scEta;
    vals[8] = scPhi;
    vals[9] = R9;
    vals[10] = etawidth;
    vals[11] = phiwidth;
    vals[12] = NClusters;
    vals[13] = HoE;
    vals[14] = EtaSeed - scEta;
    vals[15] = atan2(sin(PhiSeed - scPhi), cos(PhiSeed - scPhi));
    vals[16] = ESeed / SCRawEnergy;
    vals[17] = E3x3Seed / ESeed;
    vals[18] = E5x5Seed / SCRawEnergy;
    vals[19] = E5x5Seed / ESeed;
    vals[20] = EMaxSeed / ESeed;
    vals[21] = E2ndSeed / ESeed;
    vals[22] = ETopSeed / ESeed;
    vals[23] = EBottomSeed / ESeed;
    vals[24] = ELeftSeed / ESeed;
    vals[25] = ERightSeed / ESeed;
    vals[26] = E2x5MaxSeed / ESeed;
    vals[27] = E2x5TopSeed / ESeed;
    vals[28] = E2x5BottomSeed / ESeed;
    vals[29] = E2x5LeftSeed / ESeed;
    vals[30] = E2x5RightSeed / ESeed;
    vals[31] = see;
    vals[32] = spp;
    vals[33] = sep;
    vals[34] = phiwidth / etawidth;
    vals[35] = (ELeftSeed + ERightSeed == 0. ? 0. : (ELeftSeed - ERightSeed) / (ELeftSeed + ERightSeed));
    vals[36] = (ETopSeed + EBottomSeed == 0. ? 0. : (ETopSeed - EBottomSeed) / (ETopSeed + EBottomSeed));
    vals[37] = ESubs / SCRawEnergy;
    vals[38] = ESub1 / SCRawEnergy;
    vals[39] = (NClusters <= 1 ? 999. : EtaSub1 - EtaSeed);
    vals[40] = (NClusters <= 1 ? 999. : atan2(sin(PhiSub1 - PhiSeed), cos(PhiSub1 - PhiSeed)));
    vals[41] = (NClusters <= 1 ? 0. : EMaxSub1 / ESub1);
    vals[42] = (NClusters <= 1 ? 0. : E3x3Sub1 / ESub1);
    vals[43] = ESub2 / SCRawEnergy;
    vals[44] = (NClusters <= 2 ? 999. : EtaSub2 - EtaSeed);
    vals[45] = (NClusters <= 2 ? 999. : atan2(sin(PhiSub2 - PhiSeed), cos(PhiSub2 - PhiSeed)));
    vals[46] = (NClusters <= 2 ? 0. : EMaxSub2 / ESub2);
    vals[47] = (NClusters <= 2 ? 0. : E3x3Sub2 / ESub2);
    vals[48] = ESub3 / SCRawEnergy;
    vals[49] = (NClusters <= 3 ? 999. : EtaSub3 - EtaSeed);
    vals[50] = (NClusters <= 3 ? 999. : atan2(sin(PhiSub3 - PhiSeed), cos(PhiSub3 - PhiSeed)));
    vals[51] = (NClusters <= 3 ? 0. : EMaxSub3 / ESub3);
    vals[52] = (NClusters <= 3 ? 0. : E3x3Sub3 / ESub3);
    vals[53] = IEtaSeed;
    vals[54] = ((int)IEtaSeed) % 5;
    vals[55] = (std::abs(IEtaSeed) <= 25) * (((int)IEtaSeed) % 25) +
               (std::abs(IEtaSeed) > 25) * (((int)(IEtaSeed - 25 * std::abs(IEtaSeed) / IEtaSeed)) % 20);
    vals[56] = IPhiSeed;
    vals[57] = ((int)IPhiSeed) % 2;
    vals[58] = ((int)IPhiSeed) % 20;
    vals[59] = EtaCrySeed;
    vals[60] = PhiCrySeed;
  } else {  // Endcap
    vals[0] = rho;
    vals[1] = vertices;
    vals[2] = isEcalDriven;
    vals[3] = isEtaGap;
    vals[4] = isPhiGap;
    vals[5] = isDeeGap;
    vals[6] = SCRawEnergy;
    vals[7] = scEta;
    vals[8] = scPhi;
    vals[9] = R9;
    vals[10] = etawidth;
    vals[11] = phiwidth;
    vals[12] = NClusters;
    vals[13] = HoE;
    vals[14] = EtaSeed - scEta;
    vals[15] = atan2(sin(PhiSeed - scPhi), cos(PhiSeed - scPhi));
    vals[16] = ESeed / SCRawEnergy;
    vals[17] = E3x3Seed / ESeed;
    vals[18] = E5x5Seed / SCRawEnergy;
    vals[19] = E5x5Seed / ESeed;
    vals[20] = EMaxSeed / ESeed;
    vals[21] = E2ndSeed / ESeed;
    vals[22] = ETopSeed / ESeed;
    vals[23] = EBottomSeed / ESeed;
    vals[24] = ELeftSeed / ESeed;
    vals[25] = ERightSeed / ESeed;
    vals[26] = E2x5MaxSeed / ESeed;
    vals[27] = E2x5TopSeed / ESeed;
    vals[28] = E2x5BottomSeed / ESeed;
    vals[29] = E2x5LeftSeed / ESeed;
    vals[30] = E2x5RightSeed / ESeed;
    vals[31] = see;
    vals[32] = spp;
    vals[33] = sep;
    vals[34] = phiwidth / etawidth;
    vals[35] = (ELeftSeed + ERightSeed == 0. ? 0. : (ELeftSeed - ERightSeed) / (ELeftSeed + ERightSeed));
    vals[36] = (ETopSeed + EBottomSeed == 0. ? 0. : (ETopSeed - EBottomSeed) / (ETopSeed + EBottomSeed));
    vals[37] = ESubs / SCRawEnergy;
    vals[38] = ESub1 / SCRawEnergy;
    vals[39] = (NClusters <= 1 ? 999. : EtaSub1 - EtaSeed);
    vals[40] = (NClusters <= 1 ? 999. : atan2(sin(PhiSub1 - PhiSeed), cos(PhiSub1 - PhiSeed)));
    vals[41] = (NClusters <= 1 ? 0. : EMaxSub1 / ESub1);
    vals[42] = (NClusters <= 1 ? 0. : E3x3Sub1 / ESub1);
    vals[43] = ESub2 / SCRawEnergy;
    vals[44] = (NClusters <= 2 ? 999. : EtaSub2 - EtaSeed);
    vals[45] = (NClusters <= 2 ? 999. : atan2(sin(PhiSub2 - PhiSeed), cos(PhiSub2 - PhiSeed)));
    vals[46] = (NClusters <= 2 ? 0. : EMaxSub2 / ESub2);
    vals[47] = (NClusters <= 2 ? 0. : E3x3Sub2 / ESub2);
    vals[48] = ESub3 / SCRawEnergy;
    vals[49] = (NClusters <= 3 ? 999. : EtaSub3 - EtaSeed);
    vals[50] = (NClusters <= 3 ? 999. : atan2(sin(PhiSub3 - PhiSeed), cos(PhiSub3 - PhiSeed)));
    vals[51] = (NClusters <= 3 ? 0. : EMaxSub3 / ESub3);
    vals[52] = (NClusters <= 3 ? 0. : E3x3Sub3 / ESub3);
    vals[53] = PreShowerOverRaw;
    vals[54] = NPshwClusters;
    vals[55] = EPshwSubs / SCRawEnergy;
    vals[56] = EPshwSub1 / SCRawEnergy;
    vals[57] = (NPshwClusters <= 0 ? 999. : EtaPshwSub1 - EtaSeed);
    vals[58] = (NPshwClusters <= 0 ? 999. : atan2(sin(PhiPshwSub1 - PhiSeed), cos(PhiPshwSub1 - PhiSeed)));
    vals[59] = EPshwSub2 / SCRawEnergy;
    vals[60] = (NPshwClusters <= 1 ? 999. : EtaPshwSub2 - EtaSeed);
    vals[61] = (NPshwClusters <= 1 ? 999. : atan2(sin(PhiPshwSub2 - PhiSeed), cos(PhiPshwSub2 - PhiSeed)));
    vals[62] = EPshwSub3 / SCRawEnergy;
    vals[63] = (NPshwClusters <= 2 ? 999. : EtaPshwSub3 - EtaSeed);
    vals[64] = (NPshwClusters <= 2 ? 999. : atan2(sin(PhiPshwSub3 - PhiSeed), cos(PhiPshwSub3 - PhiSeed)));
  }

  // Now evaluating the regression
  double regressionResult = 0;
  Int_t BinIndex = -1;

  if (fVersionType == kWithSubCluVar) {
    if (isEB) {
      regressionResult = SCRawEnergy * forestUncertainty_eb->GetResponse(vals);
      BinIndex = 0;
    } else {
      regressionResult = (SCRawEnergy * (1 + PreShowerOverRaw)) * forestUncertainty_ee->GetResponse(vals);
      BinIndex = 1;
    }
  }

  //print debug
  if (printDebug) {
    if (isEB) {
      std::cout << "Barrel :";
      for (unsigned int v = 0; v < 38; ++v)
        std::cout << vals[v] << ", ";
      std::cout << "\n";
    } else {
      std::cout << "Endcap :";
      for (unsigned int v = 0; v < 31; ++v)
        std::cout << vals[v] << ", ";
      std::cout << "\n";
    }
    std::cout << "BinIndex : " << BinIndex << "\n";
    std::cout << "SCRawEnergy = " << SCRawEnergy << " : PreShowerOverRaw = " << PreShowerOverRaw << std::endl;
    std::cout << "regression energy uncertainty = " << regressionResult << std::endl;
  }

  // Cleaning up and returning
  delete[] vals;
  return regressionResult;
}
