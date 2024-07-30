#include "CaloExtractorByAssociator.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "CondFormats/EcalObjects/interface/EcalPFRecHitThresholds.h"
#include "CondFormats/DataRecord/interface/EcalPFRecHitThresholdsRcd.h"
#include "CondFormats/DataRecord/interface/HcalPFCutsRcd.h"
#include "CondTools/Hcal/interface/HcalPFCutsHandler.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"

using namespace edm;
using namespace std;
using namespace reco;
using namespace muonisolation;
using reco::isodeposit::Direction;

namespace {
  constexpr double dRMax_CandDep = 1.0;  //pick up candidate own deposits up to this dR if theDR_Max is smaller
}

CaloExtractorByAssociator::CaloExtractorByAssociator(const ParameterSet& par, edm::ConsumesCollector&& iC)
    : theUseEcalRecHitsFlag(par.getParameter<bool>("UseEcalRecHitsFlag")),
      theUseHcalRecHitsFlag(par.getParameter<bool>("UseHcalRecHitsFlag")),
      theUseHORecHitsFlag(par.getParameter<bool>("UseHORecHitsFlag")),
      theDepositLabel(par.getUntrackedParameter<string>("DepositLabel")),
      theDepositInstanceLabels(par.getParameter<std::vector<std::string> >("DepositInstanceLabels")),
      thePropagatorName(par.getParameter<std::string>("PropagatorName")),
      theThreshold_E(par.getParameter<double>("Threshold_E")),
      theThreshold_H(par.getParameter<double>("Threshold_H")),
      theThreshold_HO(par.getParameter<double>("Threshold_HO")),
      theMaxSeverityHB(par.getParameter<int>("MaxSeverityHB")),
      theMaxSeverityHE(par.getParameter<int>("MaxSeverityHE")),
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
      bFieldToken_(iC.esConsumes()),
      thePrintTimeReport(par.getUntrackedParameter<bool>("PrintTimeReport")) {
  ParameterSet serviceParameters = par.getParameter<ParameterSet>("ServiceParameters");
  theService = new MuonServiceProxy(serviceParameters, edm::ConsumesCollector(iC));

  //theAssociatorParameters = new TrackAssociatorParameters(par.getParameter<edm::ParameterSet>("TrackAssociatorParameters"), iC);
  theAssociatorParameters = new TrackAssociatorParameters();
  theAssociatorParameters->loadParameters(par.getParameter<edm::ParameterSet>("TrackAssociatorParameters"), iC);
  theAssociator = new TrackDetectorAssociator();

  ecalRecHitThresh_ = par.getParameter<bool>("EcalRecHitThresh");
  hcalCutsFromDB_ = par.getParameter<bool>("HcalCutsFromDB");

  caloGeomToken_ = iC.esConsumes();
  ecalPFRechitThresholdsToken_ = iC.esConsumes();
  hcalCutsToken_ = iC.esConsumes();
  hcalTopologyToken_ = iC.esConsumes();
  hcalChannelQualityToken_ = iC.esConsumes(edm::ESInputTag("", "withTopo"));
  hcalSevLvlComputerToken_ = iC.esConsumes();
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

  const EcalPFRecHitThresholds* ecalThresholds = &eventSetup.getData(ecalPFRechitThresholdsToken_);
  const HcalPFCuts* hcalCuts = &eventSetup.getData(hcalCutsToken_);
  const HcalTopology* hcalTopology_ = &eventSetup.getData(hcalTopologyToken_);
  const HcalChannelQuality* hcalChStatus_ = &eventSetup.getData(hcalChannelQualityToken_);
  const HcalSeverityLevelComputer* hcalSevLvlComputer_ = &eventSetup.getData(hcalSevLvlComputerToken_);

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

  auto const& bField = eventSetup.getData(bFieldToken_);

  reco::TransientTrack tMuon(muon, &bField);
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

  if (theUseEcalRecHitsFlag) {
    //Ecal
    auto const& caloGeom = eventSetup.getData(caloGeomToken_);
    std::vector<const EcalRecHit*>::const_iterator eHitCI = mInfo.ecalRecHits.begin();
    for (; eHitCI != mInfo.ecalRecHits.end(); ++eHitCI) {
      const EcalRecHit* eHitCPtr = *eHitCI;
      GlobalPoint eHitPos = caloGeom.getPosition(eHitCPtr->detid());
      double deltaR2 = reco::deltaR2(muon, eHitPos);
      double cosTheta = 1. / cosh(eHitPos.eta());
      double energy = eHitCPtr->energy();
      double et = energy * cosTheta;
      if (deltaR2 > std::max(dRMax_CandDep * dRMax_CandDep, theDR_Max * theDR_Max))
        continue;

      if (ecalThresholds != nullptr) {  // use thresholds from rechit
        float rhThres = (ecalThresholds != nullptr) ? (*ecalThresholds)[eHitCPtr->detid()] : 0.f;
        if (energy <= rhThres)
          continue;
      } else {  // use thresholds from config
        if (et <= theThreshold_E || energy <= 3 * noiseRecHit(eHitCPtr->detid()))
          continue;
      }

      bool vetoHit = false;
      //! first check if the hit is inside the veto cone by dR-alone
      if (deltaR2 < std::pow(theDR_Veto_E, 2)) {
        LogDebug("RecoMuon|CaloExtractorByAssociator") << " >>> Veto ECAL hit: Calo deltaR2= " << deltaR2;
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
      if (deltaR2 > std::pow(theDR_Max, 2) && !vetoHit)
        continue;

      if (vetoHit) {
        depEcal.addCandEnergy(et);
      } else {
        depEcal.addDeposit(reco::isodeposit::Direction(eHitPos.eta(), eHitPos.phi()), et);
      }
    }
  }

  if (theUseHcalRecHitsFlag) {
    //Hcal
    auto const& caloGeom = eventSetup.getData(caloGeomToken_);
    std::vector<const HBHERecHit*>::const_iterator hHitCI = mInfo.hcalRecHits.begin();
    for (; hHitCI != mInfo.hcalRecHits.end(); ++hHitCI) {
      const HBHERecHit* hHitCPtr = *hHitCI;
      GlobalPoint hHitPos = caloGeom.getPosition(hHitCPtr->detid());
      double deltaR2 = reco::deltaR2(muon, hHitPos);
      double cosTheta = 1. / cosh(hHitPos.eta());
      double energy = hHitCPtr->energy();
      double et = energy * cosTheta;
      if (deltaR2 > std::max(dRMax_CandDep * dRMax_CandDep, theDR_Max * theDR_Max))
        continue;

      // check Hcal Cuts from DB
      if (hcalCuts != nullptr) {
        const HcalPFCut* item = hcalCuts->getValues(hHitCPtr->id().rawId());
        if (energy <= item->noiseThreshold())
          continue;
      } else {
        if (et <= theThreshold_H || energy <= 3 * noiseRecHit(hHitCPtr->detid()))
          continue;
      }

      const HcalDetId hid(hHitCPtr->detid());
      DetId did = hcalTopology_->idFront(hid);
      const uint32_t flag = hHitCPtr->flags();
      const uint32_t dbflag = hcalChStatus_->getValues(did)->getValue();
      bool recovered = hcalSevLvlComputer_->recoveredRecHit(did, flag);
      int severity = hcalSevLvlComputer_->getSeverityLevel(did, flag, dbflag);

      const bool goodHB = hid.subdet() == HcalBarrel and (severity <= theMaxSeverityHB or recovered);
      const bool goodHE = hid.subdet() == HcalEndcap and (severity <= theMaxSeverityHE or recovered);
      if (!goodHB and !goodHE)
        continue;

      bool vetoHit = false;
      //! first check if the hit is inside the veto cone by dR-alone
      if (deltaR2 < std::pow(theDR_Veto_H, 2)) {
        LogDebug("RecoMuon|CaloExtractorByAssociator") << " >>> Veto HBHE hit: Calo deltaR2= " << deltaR2;
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
      if (deltaR2 > std::pow(theDR_Max, 2) && !vetoHit)
        continue;

      if (vetoHit) {
        depHcal.addCandEnergy(et);
      } else {
        depHcal.addDeposit(reco::isodeposit::Direction(hHitPos.eta(), hHitPos.phi()), et);
      }
    }
  }

  if (theUseHORecHitsFlag) {
    //HOcal
    auto const& caloGeom = eventSetup.getData(caloGeomToken_);
    std::vector<const HORecHit*>::const_iterator hoHitCI = mInfo.hoRecHits.begin();
    for (; hoHitCI != mInfo.hoRecHits.end(); ++hoHitCI) {
      const HORecHit* hoHitCPtr = *hoHitCI;
      GlobalPoint hoHitPos = caloGeom.getPosition(hoHitCPtr->detid());
      double deltaR2 = reco::deltaR2(muon, hoHitPos);
      double cosTheta = 1. / cosh(hoHitPos.eta());
      double energy = hoHitCPtr->energy();
      double et = energy * cosTheta;
      if (deltaR2 > std::max(dRMax_CandDep * dRMax_CandDep, theDR_Max * theDR_Max) ||
          !(et > theThreshold_HO && energy > 3 * noiseRecHit(hoHitCPtr->detid())))
        continue;

      bool vetoHit = false;
      //! first check if the hit is inside the veto cone by dR-alone
      if (deltaR2 < std::pow(theDR_Veto_HO, 2)) {
        LogDebug("RecoMuon|CaloExtractorByAssociator") << " >>> Veto HO hit: Calo deltaR2= " << deltaR2;
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
      if (deltaR2 > std::pow(theDR_Max, 2) && !vetoHit)
        continue;

      if (vetoHit) {
        depHOcal.addCandEnergy(et);
      } else {
        depHOcal.addDeposit(reco::isodeposit::Direction(hoHitPos.eta(), hoHitPos.phi()), et);
      }
    }
  }

  if (!theUseEcalRecHitsFlag or !theUseHcalRecHitsFlag or !theUseHORecHitsFlag) {
    //! use calo towers
    std::vector<const CaloTower*>::const_iterator calCI = mInfo.towers.begin();
    for (; calCI != mInfo.towers.end(); ++calCI) {
      const CaloTower* calCPtr = *calCI;
      double deltaR2 = reco::deltaR2(muon, *calCPtr);
      if (deltaR2 > std::max(dRMax_CandDep * dRMax_CandDep, theDR_Max * theDR_Max))
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
      double deltar2Ecal = reco::deltaR2(mInfo.trkGlobPosAtEcal, *calCPtr);
      //! first check if the tower is inside the veto cone by dR-alone
      if (deltar2Ecal < std::pow(theDR_Veto_E, 2)) {
        LogDebug("RecoMuon|CaloExtractorByAssociator") << " >>> Veto ecal tower: Calo deltaR= " << deltar2Ecal;
        LogDebug("RecoMuon|CaloExtractorByAssociator")
            << " >>> Calo eta phi ethcal: " << calCPtr->eta() << " " << calCPtr->phi() << " " << ethcal;
        vetoTowerEcal = true;
      }
      bool vetoTowerHcal = false;
      double deltar2Hcal = reco::deltaR2(mInfo.trkGlobPosAtHcal, *calCPtr);
      //! first check if the tower is inside the veto cone by dR-alone
      if (deltar2Hcal < std::pow(theDR_Veto_H, 2)) {
        LogDebug("RecoMuon|CaloExtractorByAssociator") << " >>> Veto hcal tower: Calo deltaR= " << deltar2Hcal;
        LogDebug("RecoMuon|CaloExtractorByAssociator")
            << " >>> Calo eta phi ethcal: " << calCPtr->eta() << " " << calCPtr->phi() << " " << ethcal;
        vetoTowerHcal = true;
      }
      bool vetoTowerHOCal = false;
      double deltar2HOcal = reco::deltaR2(mInfo.trkGlobPosAtHO, *calCPtr);
      //! first check if the tower is inside the veto cone by dR-alone
      if (deltar2HOcal < std::pow(theDR_Veto_HO, 2)) {
        LogDebug("RecoMuon|CaloExtractorByAssociator") << " >>> Veto HO tower: Calo deltaR= " << deltar2HOcal;
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

      if (deltaR2 > std::pow(theDR_Max, 2) && !(vetoTowerEcal || vetoTowerHcal || vetoTowerHOCal))
        continue;

      reco::isodeposit::Direction towerDir(calCPtr->eta(), calCPtr->phi());
      //! add the Et of the tower to deposits if it's not a vetoed; put into muonEnergy otherwise
      if (doEcal and !theUseEcalRecHitsFlag) {
        if (vetoTowerEcal)
          depEcal.addCandEnergy(etecal);
        else if (deltaR2 <= std::pow(theDR_Max, 2))
          depEcal.addDeposit(towerDir, etecal);
      }
      if (doHcal and !theUseHcalRecHitsFlag) {
        if (vetoTowerHcal)
          depHcal.addCandEnergy(ethcal);
        else if (deltaR2 <= std::pow(theDR_Max, 2))
          depHcal.addDeposit(towerDir, ethcal);
      }
      if (doHOcal and !theUseHORecHitsFlag) {
        if (vetoTowerHOCal)
          depHOcal.addCandEnergy(ethocal);
        else if (deltaR2 <= std::pow(theDR_Max, 2))
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
