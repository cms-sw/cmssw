#include "CaloExtractorByAssociator.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"

#include "DataFormats/Math/interface/deltaR.h"

using namespace edm;
using namespace std;
using namespace reco;
using namespace muonisolation;
using reco::isodeposit::Direction;

namespace {
  constexpr double dRMax_CandDep = 1.0;  //pick up candidate own deposits up to this dR if theDR_Max is smaller
}

CaloExtractorByAssociator::CaloExtractorByAssociator(const ParameterSet& par, edm::ConsumesCollector&& iC)
    : theUseRecHitsFlag(par.getParameter<bool>("UseRecHitsFlag")),
      theDepositLabel(par.getUntrackedParameter<string>("DepositLabel")),
      theDepositInstanceLabels(par.getParameter<std::vector<std::string> >("DepositInstanceLabels")),
      thePropagatorName(par.getParameter<std::string>("PropagatorName")),
      theThreshold_E(par.getParameter<double>("Threshold_E")),
      theThreshold_H(par.getParameter<double>("Threshold_H")),
      theThreshold_HO(par.getParameter<double>("Threshold_HO")),
      theDR_Veto_E(par.getParameter<double>("DR_Veto_E")),
      theDR_Veto_H(par.getParameter<double>("DR_Veto_H")),
      theDR_Veto_HO(par.getParameter<double>("DR_Veto_HO")),
      theCenterConeOnCalIntersection(par.getParameter<bool>("CenterConeOnCalIntersection")),
      theDR_Max(par.getParameter<double>("DR_Max")),
      theNoise_EB(par.getParameter<double>("Noise_EB")),
      theNoise_EE(par.getParameter<double>("Noise_EE")),
      theNoise_HB(par.getParameter<double>("Noise_HB")),
      theNoise_HE(par.getParameter<double>("Noise_HE")),
      theNoise_HO(par.getParameter<double>("Noise_HO")),
      theNoiseTow_EB(par.getParameter<double>("NoiseTow_EB")),
      theNoiseTow_EE(par.getParameter<double>("NoiseTow_EE")),
      theService(nullptr),
      theAssociator(nullptr),
      thePrintTimeReport(par.getUntrackedParameter<bool>("PrintTimeReport")) {
  ParameterSet serviceParameters = par.getParameter<ParameterSet>("ServiceParameters");
  theService = new MuonServiceProxy(serviceParameters, edm::ConsumesCollector(iC));

  //theAssociatorParameters = new TrackAssociatorParameters(par.getParameter<edm::ParameterSet>("TrackAssociatorParameters"), iC);
  theAssociatorParameters = new TrackAssociatorParameters();
  theAssociatorParameters->loadParameters(par.getParameter<edm::ParameterSet>("TrackAssociatorParameters"), iC);
  theAssociator = new TrackDetectorAssociator();
}

CaloExtractorByAssociator::~CaloExtractorByAssociator() {
  if (theAssociatorParameters)
    delete theAssociatorParameters;
  if (theService)
    delete theService;
  if (theAssociator)
    delete theAssociator;
}

void CaloExtractorByAssociator::fillVetos(const edm::Event& event,
                                          const edm::EventSetup& eventSetup,
                                          const TrackCollection& muons) {
  //   LogWarning("CaloExtractorByAssociator")
  //     <<"fillVetos does nothing now: IsoDeposit provides enough functionality\n"
  //     <<"to remove a deposit at/around given (eta, phi)";
}

IsoDeposit CaloExtractorByAssociator::deposit(const Event& event,
                                              const EventSetup& eventSetup,
                                              const Track& muon) const {
  IsoDeposit::Direction muonDir(muon.eta(), muon.phi());
  IsoDeposit dep(muonDir);

  //   LogWarning("CaloExtractorByAssociator")
  //     <<"single deposit is not an option here\n"
  //     <<"use ::deposits --> extract all and reweight as necessary";

  return dep;
}

//! Make separate deposits: for ECAL, HCAL, HO
std::vector<IsoDeposit> CaloExtractorByAssociator::deposits(const Event& event,
                                                            const EventSetup& eventSetup,
                                                            const Track& muon) const {
  theService->update(eventSetup);
  theAssociator->setPropagator(&*(theService->propagator(thePropagatorName)));

  //! check configuration consistency
  //! could've been made at construction stage (fix later?)
  if (theDepositInstanceLabels.size() != 3) {
    LogError("MuonIsolation") << "Configuration is inconsistent: Need 3 deposit instance labels";
  }
  if (!(theDepositInstanceLabels[0].compare(0, 1, std::string("e")) == 0) ||
      !(theDepositInstanceLabels[1].compare(0, 1, std::string("h")) == 0) ||
      !(theDepositInstanceLabels[2].compare(0, 2, std::string("ho")) == 0)) {
    LogWarning("MuonIsolation")
        << "Deposit instance labels do not look like  (e*, h*, ho*):"
        << "proceed at your own risk. The extractor interprets lab0=from ecal; lab1=from hcal; lab2=from ho";
  }

  typedef IsoDeposit::Veto Veto;
  //! this should be (eventually) set to the eta-phi of the crossing point of
  //! a straight line tangent to a muon at IP and the calorimeter
  IsoDeposit::Direction muonDir(muon.eta(), muon.phi());

  IsoDeposit depEcal(muonDir);
  IsoDeposit depHcal(muonDir);
  IsoDeposit depHOcal(muonDir);

  edm::ESHandle<MagneticField> bField;
  eventSetup.get<IdealMagneticFieldRecord>().get(bField);

  reco::TransientTrack tMuon(muon, &*bField);
  FreeTrajectoryState iFTS = tMuon.initialFreeState();
  TrackDetMatchInfo mInfo = theAssociator->associate(event, eventSetup, iFTS, *theAssociatorParameters);

  //! each deposit type veto is at the point of intersect with that detector
  depEcal.setVeto(
      Veto(reco::isodeposit::Direction(mInfo.trkGlobPosAtEcal.eta(), mInfo.trkGlobPosAtEcal.phi()), theDR_Veto_E));
  depHcal.setVeto(
      Veto(reco::isodeposit::Direction(mInfo.trkGlobPosAtHcal.eta(), mInfo.trkGlobPosAtHcal.phi()), theDR_Veto_H));
  depHOcal.setVeto(
      Veto(reco::isodeposit::Direction(mInfo.trkGlobPosAtHO.eta(), mInfo.trkGlobPosAtHO.phi()), theDR_Veto_HO));

  if (theCenterConeOnCalIntersection) {
    reco::isodeposit::Direction dirTmp = depEcal.veto().vetoDir;
    double dRtmp = depEcal.veto().dR;
    depEcal = IsoDeposit(dirTmp);
    depEcal.setVeto(Veto(dirTmp, dRtmp));

    dirTmp = depHcal.veto().vetoDir;
    dRtmp = depHcal.veto().dR;
    depHcal = IsoDeposit(dirTmp);
    depHcal.setVeto(Veto(dirTmp, dRtmp));

    dirTmp = depHOcal.veto().vetoDir;
    dRtmp = depHOcal.veto().dR;
    depHOcal = IsoDeposit(dirTmp);
    depHOcal.setVeto(Veto(dirTmp, dRtmp));
  }

  if (theUseRecHitsFlag) {
    //! do things based on rec-hits here
    //! too much copy-pasting now (refactor later?)
    edm::ESHandle<CaloGeometry> caloGeom;
    eventSetup.get<CaloGeometryRecord>().get(caloGeom);

    //Ecal
    std::vector<const EcalRecHit*>::const_iterator eHitCI = mInfo.ecalRecHits.begin();
    for (; eHitCI != mInfo.ecalRecHits.end(); ++eHitCI) {
      const EcalRecHit* eHitCPtr = *eHitCI;
      GlobalPoint eHitPos = caloGeom->getPosition(eHitCPtr->detid());
      double deltar0 = reco::deltaR(muon, eHitPos);
      double cosTheta = 1. / cosh(eHitPos.eta());
      double energy = eHitCPtr->energy();
      double et = energy * cosTheta;
      if (deltar0 > std::max(dRMax_CandDep, theDR_Max) ||
          !(et > theThreshold_E && energy > 3 * noiseRecHit(eHitCPtr->detid())))
        continue;

      bool vetoHit = false;
      double deltar = reco::deltaR(mInfo.trkGlobPosAtEcal, eHitPos);
      //! first check if the hit is inside the veto cone by dR-alone
      if (deltar < theDR_Veto_E) {
        LogDebug("RecoMuon|CaloExtractorByAssociator") << " >>> Veto ECAL hit: Calo deltaR= " << deltar;
        LogDebug("RecoMuon|CaloExtractorByAssociator")
            << " >>> Calo eta phi ethcal: " << eHitPos.eta() << " " << eHitPos.phi() << " " << et;
        vetoHit = true;
      }
      //! and now pitch those in the crossed list
      if (!vetoHit) {
        for (unsigned int iH = 0; iH < mInfo.crossedEcalIds.size() && !vetoHit; ++iH) {
          if (mInfo.crossedEcalIds[iH].rawId() == eHitCPtr->detid().rawId())
            vetoHit = true;
        }
      }

      //check theDR_Max only here to keep vetoHits being added to the veto energy
      if (deltar0 > theDR_Max && !vetoHit)
        continue;

      if (vetoHit) {
        depEcal.addCandEnergy(et);
      } else {
        depEcal.addDeposit(reco::isodeposit::Direction(eHitPos.eta(), eHitPos.phi()), et);
      }
    }

    //Hcal
    std::vector<const HBHERecHit*>::const_iterator hHitCI = mInfo.hcalRecHits.begin();
    for (; hHitCI != mInfo.hcalRecHits.end(); ++hHitCI) {
      const HBHERecHit* hHitCPtr = *hHitCI;
      GlobalPoint hHitPos = caloGeom->getPosition(hHitCPtr->detid());
      double deltar0 = reco::deltaR(muon, hHitPos);
      double cosTheta = 1. / cosh(hHitPos.eta());
      double energy = hHitCPtr->energy();
      double et = energy * cosTheta;
      if (deltar0 > std::max(dRMax_CandDep, theDR_Max) ||
          !(et > theThreshold_H && energy > 3 * noiseRecHit(hHitCPtr->detid())))
        continue;

      bool vetoHit = false;
      double deltar = reco::deltaR(mInfo.trkGlobPosAtHcal, hHitPos);
      //! first check if the hit is inside the veto cone by dR-alone
      if (deltar < theDR_Veto_H) {
        LogDebug("RecoMuon|CaloExtractorByAssociator") << " >>> Veto HBHE hit: Calo deltaR= " << deltar;
        LogDebug("RecoMuon|CaloExtractorByAssociator")
            << " >>> Calo eta phi ethcal: " << hHitPos.eta() << " " << hHitPos.phi() << " " << et;
        vetoHit = true;
      }
      //! and now pitch those in the crossed list
      if (!vetoHit) {
        for (unsigned int iH = 0; iH < mInfo.crossedHcalIds.size() && !vetoHit; ++iH) {
          if (mInfo.crossedHcalIds[iH].rawId() == hHitCPtr->detid().rawId())
            vetoHit = true;
        }
      }

      //check theDR_Max only here to keep vetoHits being added to the veto energy
      if (deltar0 > theDR_Max && !vetoHit)
        continue;

      if (vetoHit) {
        depHcal.addCandEnergy(et);
      } else {
        depHcal.addDeposit(reco::isodeposit::Direction(hHitPos.eta(), hHitPos.phi()), et);
      }
    }

    //HOcal
    std::vector<const HORecHit*>::const_iterator hoHitCI = mInfo.hoRecHits.begin();
    for (; hoHitCI != mInfo.hoRecHits.end(); ++hoHitCI) {
      const HORecHit* hoHitCPtr = *hoHitCI;
      GlobalPoint hoHitPos = caloGeom->getPosition(hoHitCPtr->detid());
      double deltar0 = reco::deltaR(muon, hoHitPos);
      double cosTheta = 1. / cosh(hoHitPos.eta());
      double energy = hoHitCPtr->energy();
      double et = energy * cosTheta;
      if (deltar0 > std::max(dRMax_CandDep, theDR_Max) ||
          !(et > theThreshold_HO && energy > 3 * noiseRecHit(hoHitCPtr->detid())))
        continue;

      bool vetoHit = false;
      double deltar = reco::deltaR(mInfo.trkGlobPosAtHO, hoHitPos);
      //! first check if the hit is inside the veto cone by dR-alone
      if (deltar < theDR_Veto_HO) {
        LogDebug("RecoMuon|CaloExtractorByAssociator") << " >>> Veto HO hit: Calo deltaR= " << deltar;
        LogDebug("RecoMuon|CaloExtractorByAssociator")
            << " >>> Calo eta phi ethcal: " << hoHitPos.eta() << " " << hoHitPos.phi() << " " << et;
        vetoHit = true;
      }
      //! and now pitch those in the crossed list
      if (!vetoHit) {
        for (unsigned int iH = 0; iH < mInfo.crossedHOIds.size() && !vetoHit; ++iH) {
          if (mInfo.crossedHOIds[iH].rawId() == hoHitCPtr->detid().rawId())
            vetoHit = true;
        }
      }

      //check theDR_Max only here to keep vetoHits being added to the veto energy
      if (deltar0 > theDR_Max && !vetoHit)
        continue;

      if (vetoHit) {
        depHOcal.addCandEnergy(et);
      } else {
        depHOcal.addDeposit(reco::isodeposit::Direction(hoHitPos.eta(), hoHitPos.phi()), et);
      }
    }

  } else {
    //! use calo towers
    std::vector<const CaloTower*>::const_iterator calCI = mInfo.towers.begin();
    for (; calCI != mInfo.towers.end(); ++calCI) {
      const CaloTower* calCPtr = *calCI;
      double deltar0 = reco::deltaR(muon, *calCPtr);
      if (deltar0 > std::max(dRMax_CandDep, theDR_Max))
        continue;

      //even more copy-pasting .. need to refactor
      double etecal = calCPtr->emEt();
      double eecal = calCPtr->emEnergy();
      bool doEcal = etecal > theThreshold_E && eecal > 3 * noiseEcal(*calCPtr);
      double ethcal = calCPtr->hadEt();
      double ehcal = calCPtr->hadEnergy();
      bool doHcal = ethcal > theThreshold_H && ehcal > 3 * noiseHcal(*calCPtr);
      double ethocal = calCPtr->outerEt();
      double ehocal = calCPtr->outerEnergy();
      bool doHOcal = ethocal > theThreshold_HO && ehocal > 3 * noiseHOcal(*calCPtr);
      if ((!doEcal) && (!doHcal) && (!doHcal))
        continue;

      bool vetoTowerEcal = false;
      double deltarEcal = reco::deltaR(mInfo.trkGlobPosAtEcal, *calCPtr);
      //! first check if the tower is inside the veto cone by dR-alone
      if (deltarEcal < theDR_Veto_E) {
        LogDebug("RecoMuon|CaloExtractorByAssociator") << " >>> Veto ecal tower: Calo deltaR= " << deltarEcal;
        LogDebug("RecoMuon|CaloExtractorByAssociator")
            << " >>> Calo eta phi ethcal: " << calCPtr->eta() << " " << calCPtr->phi() << " " << ethcal;
        vetoTowerEcal = true;
      }
      bool vetoTowerHcal = false;
      double deltarHcal = reco::deltaR(mInfo.trkGlobPosAtHcal, *calCPtr);
      //! first check if the tower is inside the veto cone by dR-alone
      if (deltarHcal < theDR_Veto_H) {
        LogDebug("RecoMuon|CaloExtractorByAssociator") << " >>> Veto hcal tower: Calo deltaR= " << deltarHcal;
        LogDebug("RecoMuon|CaloExtractorByAssociator")
            << " >>> Calo eta phi ethcal: " << calCPtr->eta() << " " << calCPtr->phi() << " " << ethcal;
        vetoTowerHcal = true;
      }
      bool vetoTowerHOCal = false;
      double deltarHOcal = reco::deltaR(mInfo.trkGlobPosAtHO, *calCPtr);
      //! first check if the tower is inside the veto cone by dR-alone
      if (deltarHOcal < theDR_Veto_HO) {
        LogDebug("RecoMuon|CaloExtractorByAssociator") << " >>> Veto HO tower: Calo deltaR= " << deltarHOcal;
        LogDebug("RecoMuon|CaloExtractorByAssociator")
            << " >>> Calo eta phi ethcal: " << calCPtr->eta() << " " << calCPtr->phi() << " " << ethcal;
        vetoTowerHOCal = true;
      }

      //! and now pitch those in the crossed list
      if (!(vetoTowerHOCal && vetoTowerHcal && vetoTowerEcal)) {
        for (unsigned int iH = 0; iH < mInfo.crossedTowerIds.size(); ++iH) {
          if (mInfo.crossedTowerIds[iH].rawId() == calCPtr->id().rawId()) {
            vetoTowerEcal = true;
            vetoTowerHcal = true;
            vetoTowerHOCal = true;
            break;
          }
        }
      }

      if (deltar0 > theDR_Max && !(vetoTowerEcal || vetoTowerHcal || vetoTowerHOCal))
        continue;

      reco::isodeposit::Direction towerDir(calCPtr->eta(), calCPtr->phi());
      //! add the Et of the tower to deposits if it's not a vetoed; put into muonEnergy otherwise
      if (doEcal) {
        if (vetoTowerEcal)
          depEcal.addCandEnergy(etecal);
        else if (deltar0 <= theDR_Max)
          depEcal.addDeposit(towerDir, etecal);
      }
      if (doHcal) {
        if (vetoTowerHcal)
          depHcal.addCandEnergy(ethcal);
        else if (deltar0 <= theDR_Max)
          depHcal.addDeposit(towerDir, ethcal);
      }
      if (doHOcal) {
        if (vetoTowerHOCal)
          depHOcal.addCandEnergy(ethocal);
        else if (deltar0 <= theDR_Max)
          depHOcal.addDeposit(towerDir, ethocal);
      }
    }
  }

  std::vector<IsoDeposit> resultDeps;
  resultDeps.push_back(depEcal);
  resultDeps.push_back(depHcal);
  resultDeps.push_back(depHOcal);

  return resultDeps;
}

double CaloExtractorByAssociator::noiseEcal(const CaloTower& tower) const {
  double noise = theNoiseTow_EB;
  double eta = tower.eta();
  if (fabs(eta) > 1.479)
    noise = theNoiseTow_EE;
  return noise;
}

double CaloExtractorByAssociator::noiseHcal(const CaloTower& tower) const {
  double noise = fabs(tower.eta()) > 1.479 ? theNoise_HE : theNoise_HB;
  return noise;
}

double CaloExtractorByAssociator::noiseHOcal(const CaloTower& tower) const {
  double noise = theNoise_HO;
  return noise;
}

double CaloExtractorByAssociator::noiseRecHit(const DetId& detId) const {
  double noise = 100;
  DetId::Detector det = detId.det();
  if (det == DetId::Ecal) {
    EcalSubdetector subDet = (EcalSubdetector)(detId.subdetId());
    if (subDet == EcalBarrel) {
      noise = theNoise_EB;
    } else if (subDet == EcalEndcap) {
      noise = theNoise_EE;
    }
  } else if (det == DetId::Hcal) {
    HcalSubdetector subDet = (HcalSubdetector)(detId.subdetId());
    if (subDet == HcalBarrel) {
      noise = theNoise_HB;
    } else if (subDet == HcalEndcap) {
      noise = theNoise_HE;
    } else if (subDet == HcalOuter) {
      noise = theNoise_HO;
    }
  }
  return noise;
}
