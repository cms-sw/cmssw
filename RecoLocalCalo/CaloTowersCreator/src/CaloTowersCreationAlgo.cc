#include "RecoLocalCalo/CaloTowersCreator/interface/CaloTowersCreationAlgo.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"
#include "Geometry/CaloTopology/interface/CaloTowerConstituentsMap.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Math/Interpolator.h"
#include <cmath>

//#define EDM_ML_DEBUG

CaloTowersCreationAlgo::CaloTowersCreationAlgo()
    : theEBthreshold(-1000.),
      theEEthreshold(-1000.),

      theUseEtEBTresholdFlag(false),
      theUseEtEETresholdFlag(false),
      theUseSymEBTresholdFlag(false),
      theUseSymEETresholdFlag(false),

      theHcalThreshold(-1000.),
      theHBthreshold(-1000.),
      theHBthreshold1(-1000.),
      theHBthreshold2(-1000.),
      theHESthreshold(-1000.),
      theHESthreshold1(-1000.),
      theHEDthreshold(-1000.),
      theHEDthreshold1(-1000.),
      theHOthreshold0(-1000.),
      theHOthresholdPlus1(-1000.),
      theHOthresholdMinus1(-1000.),
      theHOthresholdPlus2(-1000.),
      theHOthresholdMinus2(-1000.),
      theHF1threshold(-1000.),
      theHF2threshold(-1000.),
      theEBGrid(std::vector<double>(5, 10.)),
      theEBWeights(std::vector<double>(5, 1.)),
      theEEGrid(std::vector<double>(5, 10.)),
      theEEWeights(std::vector<double>(5, 1.)),
      theHBGrid(std::vector<double>(5, 10.)),
      theHBWeights(std::vector<double>(5, 1.)),
      theHESGrid(std::vector<double>(5, 10.)),
      theHESWeights(std::vector<double>(5, 1.)),
      theHEDGrid(std::vector<double>(5, 10.)),
      theHEDWeights(std::vector<double>(5, 1.)),
      theHOGrid(std::vector<double>(5, 10.)),
      theHOWeights(std::vector<double>(5, 1.)),
      theHF1Grid(std::vector<double>(5, 10.)),
      theHF1Weights(std::vector<double>(5, 1.)),
      theHF2Grid(std::vector<double>(5, 10.)),
      theHF2Weights(std::vector<double>(5, 1.)),
      theEBweight(1.),
      theEEweight(1.),
      theHBweight(1.),
      theHESweight(1.),
      theHEDweight(1.),
      theHOweight(1.),
      theHF1weight(1.),
      theHF2weight(1.),
      theEcutTower(-1000.),
      theEBSumThreshold(-1000.),
      theEESumThreshold(-1000.),
      theEBEScale(50.),
      theEEEScale(50.),
      theHBEScale(50.),
      theHESEScale(50.),
      theHEDEScale(50.),
      theHOEScale(50.),
      theHF1EScale(50.),
      theHF2EScale(50.),
      theHcalTopology(nullptr),
      theGeometry(nullptr),
      theTowerConstituentsMap(nullptr),
      theHcalAcceptSeverityLevel(0),
      theRecoveredHcalHitsAreUsed(false),
      theRecoveredEcalHitsAreUsed(false),
      useRejectedHitsOnly(false),
      theHcalAcceptSeverityLevelForRejectedHit(0),
      useRejectedRecoveredHcalHits(0),
      useRejectedRecoveredEcalHits(0),
      missingHcalRescaleFactorForEcal(0.),
      theHOIsUsed(true),
      // (for momentum reconstruction algorithm)
      theMomConstrMethod(0),
      theMomHBDepth(0.),
      theMomHEDepth(0.),
      theMomEBDepth(0.),
      theMomEEDepth(0.),
      theHcalPhase(0) {}

CaloTowersCreationAlgo::CaloTowersCreationAlgo(double EBthreshold,
                                               double EEthreshold,

                                               bool useEtEBTreshold,
                                               bool useEtEETreshold,
                                               bool useSymEBTreshold,
                                               bool useSymEETreshold,

                                               double HcalThreshold,
                                               double HBthreshold,
                                               double HBthreshold1,
                                               double HBthreshold2,
                                               double HESthreshold,
                                               double HESthreshold1,
                                               double HEDthreshold,
                                               double HEDthreshold1,
                                               double HOthreshold0,
                                               double HOthresholdPlus1,
                                               double HOthresholdMinus1,
                                               double HOthresholdPlus2,
                                               double HOthresholdMinus2,
                                               double HF1threshold,
                                               double HF2threshold,
                                               double EBweight,
                                               double EEweight,
                                               double HBweight,
                                               double HESweight,
                                               double HEDweight,
                                               double HOweight,
                                               double HF1weight,
                                               double HF2weight,
                                               double EcutTower,
                                               double EBSumThreshold,
                                               double EESumThreshold,
                                               bool useHO,
                                               // (momentum reconstruction algorithm)
                                               int momConstrMethod,
                                               double momHBDepth,
                                               double momHEDepth,
                                               double momEBDepth,
                                               double momEEDepth,
                                               int hcalPhase)

    : theEBthreshold(EBthreshold),
      theEEthreshold(EEthreshold),

      theUseEtEBTresholdFlag(useEtEBTreshold),
      theUseEtEETresholdFlag(useEtEETreshold),
      theUseSymEBTresholdFlag(useSymEBTreshold),
      theUseSymEETresholdFlag(useSymEETreshold),

      theHcalThreshold(HcalThreshold),
      theHBthreshold(HBthreshold),
      theHBthreshold1(HBthreshold1),
      theHBthreshold2(HBthreshold2),
      theHESthreshold(HESthreshold),
      theHESthreshold1(HESthreshold1),
      theHEDthreshold(HEDthreshold),
      theHEDthreshold1(HEDthreshold1),
      theHOthreshold0(HOthreshold0),
      theHOthresholdPlus1(HOthresholdPlus1),
      theHOthresholdMinus1(HOthresholdMinus1),
      theHOthresholdPlus2(HOthresholdPlus2),
      theHOthresholdMinus2(HOthresholdMinus2),
      theHF1threshold(HF1threshold),
      theHF2threshold(HF2threshold),
      theEBGrid(std::vector<double>(5, 10.)),
      theEBWeights(std::vector<double>(5, 1.)),
      theEEGrid(std::vector<double>(5, 10.)),
      theEEWeights(std::vector<double>(5, 1.)),
      theHBGrid(std::vector<double>(5, 10.)),
      theHBWeights(std::vector<double>(5, 1.)),
      theHESGrid(std::vector<double>(5, 10.)),
      theHESWeights(std::vector<double>(5, 1.)),
      theHEDGrid(std::vector<double>(5, 10.)),
      theHEDWeights(std::vector<double>(5, 1.)),
      theHOGrid(std::vector<double>(5, 10.)),
      theHOWeights(std::vector<double>(5, 1.)),
      theHF1Grid(std::vector<double>(5, 10.)),
      theHF1Weights(std::vector<double>(5, 1.)),
      theHF2Grid(std::vector<double>(5, 10.)),
      theHF2Weights(std::vector<double>(5, 1.)),
      theEBweight(EBweight),
      theEEweight(EEweight),
      theHBweight(HBweight),
      theHESweight(HESweight),
      theHEDweight(HEDweight),
      theHOweight(HOweight),
      theHF1weight(HF1weight),
      theHF2weight(HF2weight),
      theEcutTower(EcutTower),
      theEBSumThreshold(EBSumThreshold),
      theEESumThreshold(EESumThreshold),
      theEBEScale(50.),
      theEEEScale(50.),
      theHBEScale(50.),
      theHESEScale(50.),
      theHEDEScale(50.),
      theHOEScale(50.),
      theHF1EScale(50.),
      theHF2EScale(50.),
      theHcalTopology(nullptr),
      theGeometry(nullptr),
      theTowerConstituentsMap(nullptr),
      theHcalAcceptSeverityLevel(0),
      theRecoveredHcalHitsAreUsed(false),
      theRecoveredEcalHitsAreUsed(false),
      useRejectedHitsOnly(false),
      theHcalAcceptSeverityLevelForRejectedHit(0),
      useRejectedRecoveredHcalHits(0),
      useRejectedRecoveredEcalHits(0),
      missingHcalRescaleFactorForEcal(0.),
      theHOIsUsed(useHO),
      // (momentum reconstruction algorithm)
      theMomConstrMethod(momConstrMethod),
      theMomHBDepth(momHBDepth),
      theMomHEDepth(momHEDepth),
      theMomEBDepth(momEBDepth),
      theMomEEDepth(momEEDepth),
      theHcalPhase(hcalPhase) {}

CaloTowersCreationAlgo::CaloTowersCreationAlgo(double EBthreshold,
                                               double EEthreshold,
                                               bool useEtEBTreshold,
                                               bool useEtEETreshold,
                                               bool useSymEBTreshold,
                                               bool useSymEETreshold,

                                               double HcalThreshold,
                                               double HBthreshold,
                                               double HBthreshold1,
                                               double HBthreshold2,
                                               double HESthreshold,
                                               double HESthreshold1,
                                               double HEDthreshold,
                                               double HEDthreshold1,
                                               double HOthreshold0,
                                               double HOthresholdPlus1,
                                               double HOthresholdMinus1,
                                               double HOthresholdPlus2,
                                               double HOthresholdMinus2,
                                               double HF1threshold,
                                               double HF2threshold,
                                               const std::vector<double>& EBGrid,
                                               const std::vector<double>& EBWeights,
                                               const std::vector<double>& EEGrid,
                                               const std::vector<double>& EEWeights,
                                               const std::vector<double>& HBGrid,
                                               const std::vector<double>& HBWeights,
                                               const std::vector<double>& HESGrid,
                                               const std::vector<double>& HESWeights,
                                               const std::vector<double>& HEDGrid,
                                               const std::vector<double>& HEDWeights,
                                               const std::vector<double>& HOGrid,
                                               const std::vector<double>& HOWeights,
                                               const std::vector<double>& HF1Grid,
                                               const std::vector<double>& HF1Weights,
                                               const std::vector<double>& HF2Grid,
                                               const std::vector<double>& HF2Weights,
                                               double EBweight,
                                               double EEweight,
                                               double HBweight,
                                               double HESweight,
                                               double HEDweight,
                                               double HOweight,
                                               double HF1weight,
                                               double HF2weight,
                                               double EcutTower,
                                               double EBSumThreshold,
                                               double EESumThreshold,
                                               bool useHO,
                                               // (for the momentum construction algorithm)
                                               int momConstrMethod,
                                               double momHBDepth,
                                               double momHEDepth,
                                               double momEBDepth,
                                               double momEEDepth,
                                               int hcalPhase)

    : theEBthreshold(EBthreshold),
      theEEthreshold(EEthreshold),

      theUseEtEBTresholdFlag(useEtEBTreshold),
      theUseEtEETresholdFlag(useEtEETreshold),
      theUseSymEBTresholdFlag(useSymEBTreshold),
      theUseSymEETresholdFlag(useSymEETreshold),

      theHcalThreshold(HcalThreshold),
      theHBthreshold(HBthreshold),
      theHBthreshold1(HBthreshold1),
      theHBthreshold2(HBthreshold2),
      theHESthreshold(HESthreshold),
      theHESthreshold1(HESthreshold1),
      theHEDthreshold(HEDthreshold),
      theHEDthreshold1(HEDthreshold1),
      theHOthreshold0(HOthreshold0),
      theHOthresholdPlus1(HOthresholdPlus1),
      theHOthresholdMinus1(HOthresholdMinus1),
      theHOthresholdPlus2(HOthresholdPlus2),
      theHOthresholdMinus2(HOthresholdMinus2),
      theHF1threshold(HF1threshold),
      theHF2threshold(HF2threshold),
      theEBGrid(EBGrid),
      theEBWeights(EBWeights),
      theEEGrid(EEGrid),
      theEEWeights(EEWeights),
      theHBGrid(HBGrid),
      theHBWeights(HBWeights),
      theHESGrid(HESGrid),
      theHESWeights(HESWeights),
      theHEDGrid(HEDGrid),
      theHEDWeights(HEDWeights),
      theHOGrid(HOGrid),
      theHOWeights(HOWeights),
      theHF1Grid(HF1Grid),
      theHF1Weights(HF1Weights),
      theHF2Grid(HF2Grid),
      theHF2Weights(HF2Weights),
      theEBweight(EBweight),
      theEEweight(EEweight),
      theHBweight(HBweight),
      theHESweight(HESweight),
      theHEDweight(HEDweight),
      theHOweight(HOweight),
      theHF1weight(HF1weight),
      theHF2weight(HF2weight),
      theEcutTower(EcutTower),
      theEBSumThreshold(EBSumThreshold),
      theEESumThreshold(EESumThreshold),
      theEBEScale(50.),
      theEEEScale(50.),
      theHBEScale(50.),
      theHESEScale(50.),
      theHEDEScale(50.),
      theHOEScale(50.),
      theHF1EScale(50.),
      theHF2EScale(50.),
      theHcalTopology(nullptr),
      theGeometry(nullptr),
      theTowerConstituentsMap(nullptr),
      theHcalAcceptSeverityLevel(0),
      theRecoveredHcalHitsAreUsed(false),
      theRecoveredEcalHitsAreUsed(false),
      useRejectedHitsOnly(false),
      theHcalAcceptSeverityLevelForRejectedHit(0),
      useRejectedRecoveredHcalHits(0),
      useRejectedRecoveredEcalHits(0),
      missingHcalRescaleFactorForEcal(0.),
      theHOIsUsed(useHO),
      // (momentum reconstruction algorithm)
      theMomConstrMethod(momConstrMethod),
      theMomHBDepth(momHBDepth),
      theMomHEDepth(momHEDepth),
      theMomEBDepth(momEBDepth),
      theMomEEDepth(momEEDepth),
      theHcalPhase(hcalPhase) {
  // static int N = 0;
  // std::cout << "VI Algo " << ++N << std::endl;
  // nalgo=N;
}

void CaloTowersCreationAlgo::setGeometry(const CaloTowerTopology* cttopo,
                                         const CaloTowerConstituentsMap* ctmap,
                                         const HcalTopology* htopo,
                                         const CaloGeometry* geo) {
  theTowerTopology = cttopo;
  theTowerConstituentsMap = ctmap;
  theHcalTopology = htopo;
  theGeometry = geo;
  theTowerGeometry = geo->getSubdetectorGeometry(DetId::Calo, CaloTowerDetId::SubdetId);

  //initialize ecal bad channel map
  ecalBadChs.resize(theTowerTopology->sizeForDenseIndexing(), 0);
}

void CaloTowersCreationAlgo::begin() {
  theTowerMap.clear();
  theTowerMapSize = 0;
  //hcalDropChMap.clear();
}

void CaloTowersCreationAlgo::process(const HBHERecHitCollection& hbhe) {
  for (HBHERecHitCollection::const_iterator hbheItr = hbhe.begin(); hbheItr != hbhe.end(); ++hbheItr)
    assignHitHcal(&(*hbheItr));
}

void CaloTowersCreationAlgo::process(const HORecHitCollection& ho) {
  for (HORecHitCollection::const_iterator hoItr = ho.begin(); hoItr != ho.end(); ++hoItr)
    assignHitHcal(&(*hoItr));
}

void CaloTowersCreationAlgo::process(const HFRecHitCollection& hf) {
  for (HFRecHitCollection::const_iterator hfItr = hf.begin(); hfItr != hf.end(); ++hfItr)
    assignHitHcal(&(*hfItr));
}

void CaloTowersCreationAlgo::process(const EcalRecHitCollection& ec) {
  for (EcalRecHitCollection::const_iterator ecItr = ec.begin(); ecItr != ec.end(); ++ecItr)
    assignHitEcal(&(*ecItr));
}

// this method should not be used any more as the towers in the changed format
// can not be properly rescaled with the "rescale" method.
// "rescale was replaced by "rescaleTowers"
//
void CaloTowersCreationAlgo::process(const CaloTowerCollection& ctc) {
  for (CaloTowerCollection::const_iterator ctcItr = ctc.begin(); ctcItr != ctc.end(); ++ctcItr) {
    rescale(&(*ctcItr));
  }
}

void CaloTowersCreationAlgo::finish(CaloTowerCollection& result) {
  // now copy this map into the final collection
  result.reserve(theTowerMapSize);
  // auto k=0U;
  //  if (!theEbHandle.isValid()) std::cout << "VI ebHandle not valid" << std::endl;
  // if (!theEeHandle.isValid()) std::cout << "VI eeHandle not valid" << std::endl;

  for (auto const& mt : theTowerMap) {
    // Convert only if there is at least one constituent in the metatower.
    // The check of constituents size in the coverted tower is still needed!
    if (!mt.empty()) {
      convert(mt.id, mt, result);
    }  // ++k;}
  }

  // assert(k==theTowerMapSize);
  // std::cout << "VI TowerMap " << theTowerMapSize << " " << k << std::endl;

  theTowerMap.clear();  // save the memory
  theTowerMapSize = 0;
}

void CaloTowersCreationAlgo::rescaleTowers(const CaloTowerCollection& ctc, CaloTowerCollection& ctcResult) {
  for (CaloTowerCollection::const_iterator ctcItr = ctc.begin(); ctcItr != ctc.end(); ++ctcItr) {
    CaloTowerDetId twrId = ctcItr->id();
    double newE_em = ctcItr->emEnergy();
    double newE_had = ctcItr->hadEnergy();
    double newE_outer = ctcItr->outerEnergy();

    double threshold = 0.0;  // not used: we do not change thresholds
    double weight = 1.0;

    // HF
    if (ctcItr->ietaAbs() >= theTowerTopology->firstHFRing()) {
      double E_short = 0.5 * newE_had;           // from the definitions for HF
      double E_long = newE_em + 0.5 * newE_had;  //
      // scale
      E_long *= theHF1weight;
      E_short *= theHF2weight;
      // convert
      newE_em = E_long - E_short;
      newE_had = 2.0 * E_short;
    }

    else {  // barrel/endcap

      // find if its in EB, or EE; determine from first ecal constituent found
      for (unsigned int iConst = 0; iConst < ctcItr->constituentsSize(); ++iConst) {
        DetId constId = ctcItr->constituent(iConst);
        if (constId.det() != DetId::Ecal)
          continue;
        getThresholdAndWeight(constId, threshold, weight);
        newE_em *= weight;
        break;
      }
      // HO
      for (unsigned int iConst = 0; iConst < ctcItr->constituentsSize(); ++iConst) {
        DetId constId = ctcItr->constituent(iConst);
        if (constId.det() != DetId::Hcal)
          continue;
        if (HcalDetId(constId).subdet() != HcalOuter)
          continue;
        getThresholdAndWeight(constId, threshold, weight);
        newE_outer *= weight;
        break;
      }
      // HB/HE
      for (unsigned int iConst = 0; iConst < ctcItr->constituentsSize(); ++iConst) {
        DetId constId = ctcItr->constituent(iConst);
        if (constId.det() != DetId::Hcal)
          continue;
        if (HcalDetId(constId).subdet() == HcalOuter)
          continue;
        getThresholdAndWeight(constId, threshold, weight);
        newE_had *= weight;
        if (ctcItr->ietaAbs() > theTowerTopology->firstHERing())
          newE_outer *= weight;
        break;
      }

    }  // barrel/endcap region

    // now make the new tower

    double newE_hadTot =
        (theHOIsUsed && twrId.ietaAbs() <= theTowerTopology->lastHORing()) ? newE_had + newE_outer : newE_had;

    GlobalPoint emPoint = ctcItr->emPosition();
    GlobalPoint hadPoint = ctcItr->emPosition();

    double f_em = 1.0 / cosh(emPoint.eta());
    double f_had = 1.0 / cosh(hadPoint.eta());

    CaloTower::PolarLorentzVector towerP4;

    if (ctcItr->ietaAbs() < theTowerTopology->firstHFRing()) {
      if (newE_em > 0)
        towerP4 += CaloTower::PolarLorentzVector(newE_em * f_em, emPoint.eta(), emPoint.phi(), 0);
      if (newE_hadTot > 0)
        towerP4 += CaloTower::PolarLorentzVector(newE_hadTot * f_had, hadPoint.eta(), hadPoint.phi(), 0);
    } else {
      double newE_tot = newE_em + newE_had;
      // for HF we use common point for ecal, hcal shower positions regardless of the method
      if (newE_tot > 0)
        towerP4 += CaloTower::PolarLorentzVector(newE_tot * f_had, hadPoint.eta(), hadPoint.phi(), 0);
    }

    CaloTower rescaledTower(twrId, newE_em, newE_had, newE_outer, -1, -1, towerP4, emPoint, hadPoint);
    // copy the timings, have to convert back to int, 1 unit = 0.01 ns
    rescaledTower.setEcalTime(int(ctcItr->ecalTime() * 100.0 + 0.5));
    rescaledTower.setHcalTime(int(ctcItr->hcalTime() * 100.0 + 0.5));
    //add topology info
    rescaledTower.setHcalSubdet(theTowerTopology->lastHBRing(),
                                theTowerTopology->lastHERing(),
                                theTowerTopology->lastHFRing(),
                                theTowerTopology->lastHORing());

    std::vector<DetId> contains;
    for (unsigned int iConst = 0; iConst < ctcItr->constituentsSize(); ++iConst) {
      contains.push_back(ctcItr->constituent(iConst));
    }
    rescaledTower.addConstituents(contains);

    rescaledTower.setCaloTowerStatus(ctcItr->towerStatusWord());

    ctcResult.push_back(rescaledTower);

  }  // end of loop over towers
}

void CaloTowersCreationAlgo::assignHitHcal(const CaloRecHit* recHit) {
  DetId detId = recHit->detid();
  DetId detIdF(detId);
  if (detId.det() == DetId::Hcal && theHcalTopology->getMergePositionFlag()) {
    detIdF = theHcalTopology->idFront(HcalDetId(detId));
#ifdef EDM_ML_DEBUG
    std::cout << "AssignHitHcal DetId " << HcalDetId(detId) << " Front " << HcalDetId(detIdF) << std::endl;
#endif
  }

  unsigned int chStatusForCT = hcalChanStatusForCaloTower(recHit);

  // this is for skipping channls: mostly needed for the creation of
  // bad towers from hits i the bad channel collections.
  if (chStatusForCT == CaloTowersCreationAlgo::IgnoredChan)
    return;

  double threshold, weight;
  getThresholdAndWeight(detId, threshold, weight);

  double energy = recHit->energy();  // original RecHit energy is used to apply thresholds
  double e = energy * weight;        // energies scaled by user weight: used in energy assignments

  // SPECIAL handling of tower 28 merged depths --> half into tower 28 and half into tower 29
  bool merge(false);
  if (detIdF.det() == DetId::Hcal && HcalDetId(detIdF).subdet() == HcalEndcap &&
      (theHcalPhase == 0 || theHcalPhase == 1) &&
      //HcalDetId(detId).depth()==3 &&
      HcalDetId(detIdF).ietaAbs() == theHcalTopology->lastHERing() - 1) {
    merge = theHcalTopology->mergedDepth29(HcalDetId(detIdF));
#ifdef EDM_ML_DEBUG
    std::cout << "Merge " << HcalDetId(detIdF) << ":" << merge << std::endl;
#endif
  }
  if (merge) {
    //////////////////////////////    unsigned int chStatusForCT = hcalChanStatusForCaloTower(recHit);

    // bad channels are counted regardless of energy threshold

    if (chStatusForCT == CaloTowersCreationAlgo::BadChan) {
      CaloTowerDetId towerDetId = theTowerConstituentsMap->towerOf(detId);
      if (towerDetId.null())
        return;
      MetaTower& tower28 = find(towerDetId);
      CaloTowerDetId towerDetId29(towerDetId.ieta() + towerDetId.zside(), towerDetId.iphi());
      MetaTower& tower29 = find(towerDetId29);
      tower28.numBadHcalCells += 1;
      tower29.numBadHcalCells += 1;
    }

    else if (0.5 * energy >= threshold) {  // not bad channel: use energy if above threshold

      CaloTowerDetId towerDetId = theTowerConstituentsMap->towerOf(detId);
      if (towerDetId.null())
        return;
      MetaTower& tower28 = find(towerDetId);
      CaloTowerDetId towerDetId29(towerDetId.ieta() + towerDetId.zside(), towerDetId.iphi());
      MetaTower& tower29 = find(towerDetId29);

      if (chStatusForCT == CaloTowersCreationAlgo::RecoveredChan) {
        tower28.numRecHcalCells += 1;
        tower29.numRecHcalCells += 1;
      } else if (chStatusForCT == CaloTowersCreationAlgo::ProblematicChan) {
        tower28.numProbHcalCells += 1;
        tower29.numProbHcalCells += 1;
      }

      // NOTE DIVIDE BY 2!!!
      double e28 = 0.5 * e;
      double e29 = 0.5 * e;

      tower28.E_had += e28;
      tower28.E += e28;
      std::pair<DetId, float> mc(detId, e28);
      tower28.metaConstituents.push_back(mc);

      tower29.E_had += e29;
      tower29.E += e29;
      tower29.metaConstituents.push_back(mc);

      // time info: do not use in averaging if timing error is found: need
      // full set of status info to implement: use only "good" channels for now

      if (chStatusForCT == CaloTowersCreationAlgo::GoodChan) {
        tower28.hadSumTimeTimesE += (e28 * recHit->time());
        tower28.hadSumEForTime += e28;
        tower29.hadSumTimeTimesE += (e29 * recHit->time());
        tower29.hadSumEForTime += e29;
      }

      // store the energy in layer 3 also in E_outer
      tower28.E_outer += e28;
      tower29.E_outer += e29;
    }  // not a "bad" hit
  }    // end of special case

  else {
    HcalDetId hcalDetId(detId);

    ///////////////////////      unsigned int chStatusForCT = hcalChanStatusForCaloTower(recHit);

    if (hcalDetId.subdet() == HcalOuter) {
      CaloTowerDetId towerDetId = theTowerConstituentsMap->towerOf(detId);
      if (towerDetId.null())
        return;
      MetaTower& tower = find(towerDetId);

      if (chStatusForCT == CaloTowersCreationAlgo::BadChan) {
        if (theHOIsUsed)
          tower.numBadHcalCells += 1;
      }

      else if (energy >= threshold) {
        tower.E_outer += e;  // store HO energy even if HO is not used
        // add energy of the tower and/or flag if theHOIsUsed
        if (theHOIsUsed) {
          tower.E += e;

          if (chStatusForCT == CaloTowersCreationAlgo::RecoveredChan) {
            tower.numRecHcalCells += 1;
          } else if (chStatusForCT == CaloTowersCreationAlgo::ProblematicChan) {
            tower.numProbHcalCells += 1;
          }
        }  // HO is used

        // add HO to constituents even if it is not used: JetMET wants to keep these towers
        std::pair<DetId, float> mc(detId, e);
        tower.metaConstituents.push_back(mc);

      }  // not a bad channel, energy above threshold

    }  // HO hit

    // HF calculates EM fraction differently
    else if (hcalDetId.subdet() == HcalForward) {
      if (chStatusForCT == CaloTowersCreationAlgo::BadChan) {
        CaloTowerDetId towerDetId = theTowerConstituentsMap->towerOf(detId);
        if (towerDetId.null())
          return;
        MetaTower& tower = find(towerDetId);
        tower.numBadHcalCells += 1;
      }

      else if (energy >= threshold) {
        CaloTowerDetId towerDetId = theTowerConstituentsMap->towerOf(detId);
        if (towerDetId.null())
          return;
        MetaTower& tower = find(towerDetId);

        if (hcalDetId.depth() == 1) {
          // long fiber, so E_EM = E(Long) - E(Short)
          tower.E_em += e;
        } else {
          // short fiber, EHAD = 2 * E(Short)
          tower.E_em -= e;
          tower.E_had += 2. * e;
        }
        tower.E += e;
        if (chStatusForCT == CaloTowersCreationAlgo::RecoveredChan) {
          tower.numRecHcalCells += 1;
        } else if (chStatusForCT == CaloTowersCreationAlgo::ProblematicChan) {
          tower.numProbHcalCells += 1;
        }

        // put the timing in HCAL -> have to check timing errors when available
        // for now use only good channels
        if (chStatusForCT == CaloTowersCreationAlgo::GoodChan) {
          tower.hadSumTimeTimesE += (e * recHit->time());
          tower.hadSumEForTime += e;
        }

        std::pair<DetId, float> mc(detId, e);
        tower.metaConstituents.push_back(mc);

      }  // not a bad HF channel, energy above threshold

    }  // HF hit

    else {
      // HCAL situation normal in HB/HE
      if (chStatusForCT == CaloTowersCreationAlgo::BadChan) {
        CaloTowerDetId towerDetId = theTowerConstituentsMap->towerOf(detId);
        if (towerDetId.null())
          return;
        MetaTower& tower = find(towerDetId);
        tower.numBadHcalCells += 1;
      } else if (energy >= threshold) {
        CaloTowerDetId towerDetId = theTowerConstituentsMap->towerOf(detId);
        if (towerDetId.null())
          return;
        MetaTower& tower = find(towerDetId);
        tower.E_had += e;
        tower.E += e;
        if (chStatusForCT == CaloTowersCreationAlgo::RecoveredChan) {
          tower.numRecHcalCells += 1;
        } else if (chStatusForCT == CaloTowersCreationAlgo::ProblematicChan) {
          tower.numProbHcalCells += 1;
        }

        // Timing information: need specific accessors
        // for now use only good channels
        if (chStatusForCT == CaloTowersCreationAlgo::GoodChan) {
          tower.hadSumTimeTimesE += (e * recHit->time());
          tower.hadSumEForTime += e;
        }
        // store energy in highest depth for towers 18-27 (for electron,photon ID in endcap)
        // also, store energy in HE part of tower 16 (for JetMET cleanup)
        HcalDetId hcalDetId(detId);
        if (hcalDetId.subdet() == HcalEndcap) {
          if (theHcalPhase == 0) {
            if ((hcalDetId.depth() == 2 && hcalDetId.ietaAbs() >= 18 && hcalDetId.ietaAbs() < 27) ||
                (hcalDetId.depth() == 3 && hcalDetId.ietaAbs() == 27) ||
                (hcalDetId.depth() == 3 && hcalDetId.ietaAbs() == 16)) {
              tower.E_outer += e;
            }
          }
          //combine depths in phase0-like way
          else if (theHcalPhase == 1) {
            if ((hcalDetId.depth() >= 3 && hcalDetId.ietaAbs() >= 18 && hcalDetId.ietaAbs() < 26) ||
                (hcalDetId.depth() >= 4 && (hcalDetId.ietaAbs() == 26 || hcalDetId.ietaAbs() == 27)) ||
                (hcalDetId.depth() == 3 && hcalDetId.ietaAbs() == 17) ||
                (hcalDetId.depth() == 4 && hcalDetId.ietaAbs() == 16)) {
              tower.E_outer += e;
            }
          }
        }

        std::pair<DetId, float> mc(detId, e);
        tower.metaConstituents.push_back(mc);

      }  // not a "bad" channel, energy above threshold

    }  // channel in HBHE (excluding twrs 28,29)

  }  // recHit normal case (not in HE towers 28,29)

}  // end of assignHitHcal method

void CaloTowersCreationAlgo::assignHitEcal(const EcalRecHit* recHit) {
  DetId detId = recHit->detid();

  unsigned int chStatusForCT;
  bool ecalIsBad = false;
  std::tie(chStatusForCT, ecalIsBad) = ecalChanStatusForCaloTower(recHit);

  // this is for skipping channls: mostly needed for the creation of
  // bad towers from hits i the bad channel collections.
  if (chStatusForCT == CaloTowersCreationAlgo::IgnoredChan)
    return;

  double threshold, weight;
  getThresholdAndWeight(detId, threshold, weight);

  double energy = recHit->energy();  // original RecHit energy is used to apply thresholds
  double e = energy * weight;        // energies scaled by user weight: used in energy assignments

  /////////////////////////////      unsigned int chStatusForCT = ecalChanStatusForCaloTower(recHit);

  // For ECAL we count all bad channels after the metatower is complete

  // Include options for symmetric thresholds and cut on Et
  // for ECAL RecHits

  bool passEmThreshold = false;

  if (detId.subdetId() == EcalBarrel) {
    if (theUseEtEBTresholdFlag)
      energy /= cosh((theGeometry->getGeometry(detId)->getPosition()).eta());
    if (theUseSymEBTresholdFlag)
      passEmThreshold = (fabs(energy) >= threshold);
    else
      passEmThreshold = (energy >= threshold);

  } else if (detId.subdetId() == EcalEndcap) {
    if (theUseEtEETresholdFlag)
      energy /= cosh((theGeometry->getGeometry(detId)->getPosition()).eta());
    if (theUseSymEETresholdFlag)
      passEmThreshold = (fabs(energy) >= threshold);
    else
      passEmThreshold = (energy >= threshold);
  }

  CaloTowerDetId towerDetId = theTowerConstituentsMap->towerOf(detId);
  if (towerDetId.null())
    return;
  MetaTower& tower = find(towerDetId);

  // count bad cells and avoid double counting with those from DB (Recovered are counted bad)

  // somehow misses some
  // if ( (chStatusForCT == CaloTowersCreationAlgo::BadChan) & (!ecalIsBad) ) ++tower.numBadEcalCells;

  // a bit slower...
  if (chStatusForCT == CaloTowersCreationAlgo::BadChan) {
    auto thisEcalSevLvl = theEcalSevLvlAlgo->severityLevel(detId);
    // check if the Ecal severity is ok to keep
    auto sevit = std::find(theEcalSeveritiesToBeExcluded.begin(), theEcalSeveritiesToBeExcluded.end(), thisEcalSevLvl);
    if (sevit == theEcalSeveritiesToBeExcluded.end())
      ++tower.numBadEcalCells;  // notinDB
  }

  //      if (chStatusForCT != CaloTowersCreationAlgo::BadChan && energy >= threshold) {
  if (chStatusForCT != CaloTowersCreationAlgo::BadChan && passEmThreshold) {
    tower.E_em += e;
    tower.E += e;

    if (chStatusForCT == CaloTowersCreationAlgo::RecoveredChan) {
      tower.numRecEcalCells += 1;
    } else if (chStatusForCT == CaloTowersCreationAlgo::ProblematicChan) {
      tower.numProbEcalCells += 1;
    }

    // change when full status info is available
    // for now use only good channels

    // add e>0 check (new options allow e<0)
    if (chStatusForCT == CaloTowersCreationAlgo::GoodChan && e > 0) {
      tower.emSumTimeTimesE += (e * recHit->time());
      tower.emSumEForTime += e;  // see above
    }

    std::pair<DetId, float> mc(detId, e);
    tower.metaConstituents.push_back(mc);
  }
}  // end of assignHitEcal method

// This method is not flexible enough for the new CaloTower format.
// For now make a quick compatibility "fix" : WILL NOT WORK CORRECTLY with anything
// except the default simple p4 assignment!!!
// Must be rewritten for full functionality.
void CaloTowersCreationAlgo::rescale(const CaloTower* ct) {
  double threshold, weight;
  CaloTowerDetId towerDetId = theTowerConstituentsMap->towerOf(ct->id());
  if (towerDetId.null())
    return;
  MetaTower& tower = find(towerDetId);

  tower.E_em = 0.;
  tower.E_had = 0.;
  tower.E_outer = 0.;
  for (unsigned int i = 0; i < ct->constituentsSize(); i++) {
    DetId detId = ct->constituent(i);
    getThresholdAndWeight(detId, threshold, weight);
    DetId::Detector det = detId.det();
    if (det == DetId::Ecal) {
      tower.E_em = ct->emEnergy() * weight;
    } else {
      HcalDetId hcalDetId(detId);
      if (hcalDetId.subdet() == HcalForward) {
        if (hcalDetId.depth() == 1)
          tower.E_em = ct->emEnergy() * weight;
        if (hcalDetId.depth() == 2)
          tower.E_had = ct->hadEnergy() * weight;
      } else if (hcalDetId.subdet() == HcalOuter) {
        tower.E_outer = ct->outerEnergy() * weight;
      } else {
        tower.E_had = ct->hadEnergy() * weight;
      }
    }
    tower.E = tower.E_had + tower.E_em + tower.E_outer;

    // this is to be compliant with the new MetaTower setup
    // used only for the default simple vector assignment
    std::pair<DetId, float> mc(detId, 0);
    tower.metaConstituents.push_back(mc);
  }

  // preserve time inforamtion
  tower.emSumTimeTimesE = ct->ecalTime();
  tower.hadSumTimeTimesE = ct->hcalTime();
  tower.emSumEForTime = 1.0;
  tower.hadSumEForTime = 1.0;
}

CaloTowersCreationAlgo::MetaTower& CaloTowersCreationAlgo::find(const CaloTowerDetId& detId) {
  if (theTowerMap.empty()) {
    theTowerMap.resize(theTowerTopology->sizeForDenseIndexing());
  }

  auto& mt = theTowerMap[theTowerTopology->denseIndex(detId)];

  if (mt.empty()) {
    mt.id = detId;
    mt.metaConstituents.reserve(detId.ietaAbs() < theTowerTopology->firstHFRing() ? 12 : 2);
    ++theTowerMapSize;
  }

  return mt;
}

void CaloTowersCreationAlgo::convert(const CaloTowerDetId& id, const MetaTower& mt, CaloTowerCollection& collection) {
  assert(id.rawId() != 0);

  double ecalThres = (id.ietaAbs() <= 17) ? (theEBSumThreshold) : (theEESumThreshold);
  double E = mt.E;
  double E_em = mt.E_em;
  double E_had = mt.E_had;
  double E_outer = mt.E_outer;

  // Note: E_outer is used to save HO energy OR energy in the outermost depths in endcap region
  // In the methods with separate treatment of EM and HAD components:
  //  - HO is not used to determine direction, however HO energy is added to get "total had energy"
  //  => Check if the tower is within HO coverage before adding E_outer to the "total had" energy
  //     else the energy will be double counted
  // When summing up the energy of the tower these checks are performed in the loops over RecHits

  std::vector<std::pair<DetId, float> > metaContains = mt.metaConstituents;
  if (id.ietaAbs() < theTowerTopology->firstHFRing() && E_em < ecalThres) {  // ignore EM threshold in HF
    E -= E_em;
    E_em = 0;
    std::vector<std::pair<DetId, float> > metaContains_noecal;

    for (std::vector<std::pair<DetId, float> >::iterator i = metaContains.begin(); i != metaContains.end(); ++i)
      if (i->first.det() != DetId::Ecal)
        metaContains_noecal.push_back(*i);
    metaContains.swap(metaContains_noecal);
  }
  if (id.ietaAbs() < theTowerTopology->firstHFRing() && E_had < theHcalThreshold) {
    E -= E_had;

    if (theHOIsUsed && id.ietaAbs() <= theTowerTopology->lastHORing())
      E -= E_outer;  // not subtracted before, think it should be done

    E_had = 0;
    E_outer = 0;
    std::vector<std::pair<DetId, float> > metaContains_nohcal;

    for (std::vector<std::pair<DetId, float> >::iterator i = metaContains.begin(); i != metaContains.end(); ++i)
      if (i->first.det() != DetId::Hcal)
        metaContains_nohcal.push_back(*i);
    metaContains.swap(metaContains_nohcal);
  }

  if (metaContains.empty())
    return;

  if (missingHcalRescaleFactorForEcal > 0 && E_had == 0 && E_em > 0) {
    auto match = hcalDropChMap.find(id);
    if (match != hcalDropChMap.end() && match->second.second) {
      E_had = missingHcalRescaleFactorForEcal * E_em;
      E += E_had;
    }
  }

  double E_had_tot = (theHOIsUsed && id.ietaAbs() <= theTowerTopology->lastHORing()) ? E_had + E_outer : E_had;

  // create CaloTower using the selected algorithm

  GlobalPoint emPoint, hadPoint;

  // this is actually a 4D vector
  Basic3DVectorF towerP4;
  bool massless = true;
  // float mass1=0;
  float mass2 = 0;

  // conditional assignment of depths for barrel/endcap
  // Some additional tuning may be required in the transitional region
  // 14<|iEta|<19
  double momEmDepth = 0.;
  double momHadDepth = 0.;
  if (id.ietaAbs() <= 17) {
    momHadDepth = theMomHBDepth;
    momEmDepth = theMomEBDepth;
  } else {
    momHadDepth = theMomHEDepth;
    momEmDepth = theMomEEDepth;
  }

  switch (theMomConstrMethod) {
      // FIXME  : move to simple cartesian algebra
    case 0: {  // Simple 4-momentum assignment
      GlobalPoint p = theTowerGeometry->getGeometry(id)->getPosition();
      towerP4 = p.basicVector().unit();
      towerP4[3] = 1.f;  // energy
      towerP4 *= E;

      // double pf=1.0/cosh(p.eta());
      // if (E>0) towerP4 = CaloTower::PolarLorentzVector(E*pf, p.eta(), p.phi(), 0);

      emPoint = p;
      hadPoint = p;
    }  // end case 0
    break;

    case 1: {  // separate 4-vectors for ECAL, HCAL, add to get the 4-vector of the tower (=>tower has mass!)
      if (id.ietaAbs() < theTowerTopology->firstHFRing()) {
        Basic3DVectorF emP4;
        if (E_em > 0) {
          emPoint = emShwrPos(metaContains, momEmDepth, E_em);
          emP4 = emPoint.basicVector().unit();
          emP4[3] = 1.f;  // energy
          towerP4 = emP4 * E_em;

          // double emPf = 1.0/cosh(emPoint.eta());
          // towerP4 += CaloTower::PolarLorentzVector(E_em*emPf, emPoint.eta(), emPoint.phi(), 0);
        }
        if ((E_had + E_outer) > 0) {
          massless = (E_em <= 0);
          hadPoint = hadShwrPos(id, momHadDepth);
          auto lP4 = hadPoint.basicVector().unit();
          lP4[3] = 1.f;  // energy
          if (!massless) {
            auto diff = lP4 - emP4;
            mass2 = std::sqrt(E_em * E_had_tot * diff.mag2());
          }
          lP4 *= E_had_tot;
          towerP4 += lP4;
          /*
            if (!massless) {
               auto p = towerP4;
               double m2 = double(p[3]*p[3]) - double(p[0]*p[0])+double(p[1]*p[1])+double(p[2]*p[2]); mass1 = m2>0 ? std::sqrt(m2) : 0;
            }	    
            */
          // double hadPf = 1.0/cosh(hadPoint.eta());
          // if (E_had_tot>0) {
          //  towerP4 += CaloTower::PolarLorentzVector(E_had_tot*hadPf, hadPoint.eta(), hadPoint.phi(), 0);
          // }
        }
      } else {  // forward detector: use the CaloTower position
        GlobalPoint p = theTowerGeometry->getGeometry(id)->getPosition();
        towerP4 = p.basicVector().unit();
        towerP4[3] = 1.f;  // energy
        towerP4 *= E;
        // double pf=1.0/cosh(p.eta());
        // if (E>0) towerP4 = CaloTower::PolarLorentzVector(E*pf, p.eta(), p.phi(), 0);  // simple momentum assignment, same position
        emPoint = p;
        hadPoint = p;
      }
    }  // end case 1
    break;

    case 2: {  // use ECAL position for the tower (when E_cal>0), else default CaloTower position (massless tower)
      if (id.ietaAbs() < theTowerTopology->firstHFRing()) {
        if (E_em > 0)
          emPoint = emShwrLogWeightPos(metaContains, momEmDepth, E_em);
        else
          emPoint = theTowerGeometry->getGeometry(id)->getPosition();
        towerP4 = emPoint.basicVector().unit();
        towerP4[3] = 1.f;  // energy
        towerP4 *= E;

        // double sumPf = 1.0/cosh(emPoint.eta());
        /// if (E>0) towerP4 = CaloTower::PolarLorentzVector(E*sumPf, emPoint.eta(), emPoint.phi(), 0);

        hadPoint = emPoint;
      } else {  // forward detector: use the CaloTower position
        GlobalPoint p = theTowerGeometry->getGeometry(id)->getPosition();
        towerP4 = p.basicVector().unit();
        towerP4[3] = 1.f;  // energy
        towerP4 *= E;

        // double pf=1.0/cosh(p.eta());
        // if (E>0) towerP4 = CaloTower::PolarLorentzVector(E*pf, p.eta(), p.phi(), 0);  // simple momentum assignment, same position
        emPoint = p;
        hadPoint = p;
      }
    }  // end case 2
    break;

  }  // end of decision on p4 reconstruction method

  // insert in collection (remove and return if below threshold)
  if UNLIKELY ((towerP4[3] == 0) & (E_outer > 0)) {
    float val = theHOIsUsed ? 0 : 1E-9;  // to keep backwards compatibility for theHOIsUsed == true
    collection.emplace_back(id,
                            E_em,
                            E_had,
                            E_outer,
                            -1,
                            -1,
                            CaloTower::PolarLorentzVector(val, hadPoint.eta(), hadPoint.phi(), 0),
                            emPoint,
                            hadPoint);
  } else {
    collection.emplace_back(
        id, E_em, E_had, E_outer, -1, -1, GlobalVector(towerP4), towerP4[3], mass2, emPoint, hadPoint);
  }
  auto& caloTower = collection.back();

  // if (!massless) std::cout << "massive " << id <<' ' << mass1 <<' ' << mass2 <<' ' <<  caloTower.mass() << std::endl;
  // std::cout << "CaloTowerVI " <<theMomConstrMethod <<' ' << id <<' '<< E_em <<' '<< E_had <<' '<< E_outer <<' '<< GlobalVector(towerP4) <<' '<< towerP4[3] <<' '<< emPoint <<' '<< hadPoint << std::endl;
  //if (towerP4[3]==0) std::cout << "CaloTowerVIzero " << theEcutTower << ' ' << collection.back().eta() <<' '<< collection.back().phi() << std::endl;

  if (caloTower.energy() < theEcutTower) {
    collection.pop_back();
    return;
  }

  // set the timings
  float ecalTime = (mt.emSumEForTime > 0) ? mt.emSumTimeTimesE / mt.emSumEForTime : -9999;
  float hcalTime = (mt.hadSumEForTime > 0) ? mt.hadSumTimeTimesE / mt.hadSumEForTime : -9999;
  caloTower.setEcalTime(compactTime(ecalTime));
  caloTower.setHcalTime(compactTime(hcalTime));
  //add topology info
  caloTower.setHcalSubdet(theTowerTopology->lastHBRing(),
                          theTowerTopology->lastHERing(),
                          theTowerTopology->lastHFRing(),
                          theTowerTopology->lastHORing());

  // set the CaloTower status word =====================================
  // Channels must be counter exclusively in the defined cathegories
  // "Bad" channels (not used in energy assignment) can be flagged during
  // CaloTower creation only if specified in the configuration file

  unsigned int numBadHcalChan = mt.numBadHcalCells;
  //    unsigned int numBadEcalChan  = mt.numBadEcalCells;
  unsigned int numBadEcalChan = 0;  //

  unsigned int numRecHcalChan = mt.numRecHcalCells;
  unsigned int numRecEcalChan = mt.numRecEcalCells;
  unsigned int numProbHcalChan = mt.numProbHcalCells;
  unsigned int numProbEcalChan = mt.numProbEcalCells;

  // now add dead/off/... channels not used in RecHit reconstruction for HCAL
  HcalDropChMap::iterator dropChItr = hcalDropChMap.find(id);
  if (dropChItr != hcalDropChMap.end())
    numBadHcalChan += dropChItr->second.first;

  // for ECAL the number of all bad channels is obtained here -----------------------

  /*
    // old hyper slow algorithm    
    // get all possible constituents of the tower
    std::vector<DetId> allConstituents = theTowerConstituentsMap->constituentsOf(id);

    for (std::vector<DetId>::iterator ac_it=allConstituents.begin(); 
	 ac_it!=allConstituents.end(); ++ac_it) {

      if (ac_it->det()!=DetId::Ecal) continue;
 
      int thisEcalSevLvl = -999;
     
      if (ac_it->subdetId() == EcalBarrel && theEbHandle.isValid()) {
	thisEcalSevLvl = theEcalSevLvlAlgo->severityLevel( *ac_it, *theEbHandle);//, *theEcalChStatus);
      }
      else if (ac_it->subdetId() == EcalEndcap && theEeHandle.isValid()) {
	thisEcalSevLvl = theEcalSevLvlAlgo->severityLevel( *ac_it, *theEeHandle);//, *theEcalChStatus);
      }
 
      // check if the Ecal severity is ok to keep
      std::vector<int>::const_iterator sevit = std::find(theEcalSeveritiesToBeExcluded.begin(),
							 theEcalSeveritiesToBeExcluded.end(),
							 thisEcalSevLvl);
      if (sevit!=theEcalSeveritiesToBeExcluded.end()) {
	++numBadEcalChan;
      }

     }
     
     // compare with fast version
 
   //  hcal: 
    int inEcals[2] = {0,0};
    for (std::vector<std::pair<DetId,float> >::iterator i=metaContains.begin(); i!=metaContains.end(); ++i) {
      DetId detId = i->first;
      if(detId.det() == DetId::Ecal){
	if( detId.subdetId()==EcalBarrel ) inEcals[0] =1;
	else if( detId.subdetId()==EcalEndcap ) inEcals[1] =1;
      }
    }

     auto  numBadEcalChanNew = ecalBadChs[theTowerTopology->denseIndex(id)]+mt.numBadEcalCells; // - mt.numRecEcalCells
     if (int(numBadEcalChanNew)!=int(numBadEcalChan)) { 
       std::cout << "VI wrong " << ((inEcals[1]==1) ? "EE" : "" ) << id << " " << numBadEcalChanNew << " " << numBadEcalChan 
		 << " " << mt.numBadEcalCells << " " << mt.numRecEcalCells << std::endl;
     }
     */

  numBadEcalChan = ecalBadChs[theTowerTopology->denseIndex(id)] + mt.numBadEcalCells;  // - mt.numRecEcalCells

  //--------------------------------------------------------------------------------------

  caloTower.setCaloTowerStatus(
      numBadHcalChan, numBadEcalChan, numRecHcalChan, numRecEcalChan, numProbHcalChan, numProbEcalChan);

  double maxCellE = -999.0;  // for storing the hottest cell E in the calotower

  std::vector<DetId> contains;
  contains.reserve(metaContains.size());
  for (std::vector<std::pair<DetId, float> >::iterator i = metaContains.begin(); i != metaContains.end(); ++i) {
    contains.push_back(i->first);

    if (maxCellE < i->second) {
      // need an extra check because of the funny towers that are empty except for the presence of an HO
      // hit in the constituents (JetMET wanted them saved)
      // This constituent is only used for storing the tower, but should not be concidered as a hot cell canditate for
      // configurations with useHO = false

      if (i->first.det() == DetId::Ecal) {  // ECAL
        maxCellE = i->second;
      } else {  // HCAL
        if (HcalDetId(i->first).subdet() != HcalOuter)
          maxCellE = i->second;
        else if (theHOIsUsed)
          maxCellE = i->second;
      }

    }  // found higher E cell

  }  // loop over matacontains

  caloTower.setConstituents(std::move(contains));
  caloTower.setHottestCellE(maxCellE);

  // std::cout << "CaloTowerVI " << nalgo << ' ' << caloTower.id() << ((inEcals[1]==1) ? "EE " : " " )  << caloTower.pt() << ' ' << caloTower.et() << ' ' << caloTower.mass() << ' '
  //           << caloTower.constituentsSize() <<' '<< caloTower.towerStatusWord() << std::endl;
}

void CaloTowersCreationAlgo::getThresholdAndWeight(const DetId& detId, double& threshold, double& weight) const {
  DetId::Detector det = detId.det();
  weight = 0;  // in case the hit is not identified

  if (det == DetId::Ecal) {
    // may or may not be EB.  We'll find out.

    EcalSubdetector subdet = (EcalSubdetector)(detId.subdetId());
    if (subdet == EcalBarrel) {
      threshold = theEBthreshold;
      weight = theEBweight;
      if (weight <= 0.) {
        ROOT::Math::Interpolator my(theEBGrid, theEBWeights, ROOT::Math::Interpolation::kAKIMA);
        weight = my.Eval(theEBEScale);
      }
    } else if (subdet == EcalEndcap) {
      threshold = theEEthreshold;
      weight = theEEweight;
      if (weight <= 0.) {
        ROOT::Math::Interpolator my(theEEGrid, theEEWeights, ROOT::Math::Interpolation::kAKIMA);
        weight = my.Eval(theEEEScale);
      }
    }
  } else if (det == DetId::Hcal) {
    HcalDetId hcalDetId(detId);
    HcalSubdetector subdet = hcalDetId.subdet();
    int depth = hcalDetId.depth();

    if (subdet == HcalBarrel) {
      threshold = (depth == 1) ? theHBthreshold1 : (depth == 2) ? theHBthreshold2 : theHBthreshold;
      weight = theHBweight;
      if (weight <= 0.) {
        ROOT::Math::Interpolator my(theHBGrid, theHBWeights, ROOT::Math::Interpolation::kAKIMA);
        weight = my.Eval(theHBEScale);
      }
    }

    else if (subdet == HcalEndcap) {
      // check if it's single or double tower
      if (hcalDetId.ietaAbs() < theHcalTopology->firstHEDoublePhiRing()) {
        threshold = (depth == 1) ? theHESthreshold1 : theHESthreshold;
        weight = theHESweight;
        if (weight <= 0.) {
          ROOT::Math::Interpolator my(theHESGrid, theHESWeights, ROOT::Math::Interpolation::kAKIMA);
          weight = my.Eval(theHESEScale);
        }
      } else {
        threshold = (depth == 1) ? theHEDthreshold1 : theHEDthreshold;
        weight = theHEDweight;
        if (weight <= 0.) {
          ROOT::Math::Interpolator my(theHEDGrid, theHEDWeights, ROOT::Math::Interpolation::kAKIMA);
          weight = my.Eval(theHEDEScale);
        }
      }
    }

    else if (subdet == HcalOuter) {
      //check if it's ring 0 or +1 or +2 or -1 or -2
      if (hcalDetId.ietaAbs() <= 4)
        threshold = theHOthreshold0;
      else if (hcalDetId.ieta() < 0) {
        // set threshold for ring -1 or -2
        threshold = (hcalDetId.ietaAbs() <= 10) ? theHOthresholdMinus1 : theHOthresholdMinus2;
      } else {
        // set threshold for ring +1 or +2
        threshold = (hcalDetId.ietaAbs() <= 10) ? theHOthresholdPlus1 : theHOthresholdPlus2;
      }
      weight = theHOweight;
      if (weight <= 0.) {
        ROOT::Math::Interpolator my(theHOGrid, theHOWeights, ROOT::Math::Interpolation::kAKIMA);
        weight = my.Eval(theHOEScale);
      }
    }

    else if (subdet == HcalForward) {
      if (hcalDetId.depth() == 1) {
        threshold = theHF1threshold;
        weight = theHF1weight;
        if (weight <= 0.) {
          ROOT::Math::Interpolator my(theHF1Grid, theHF1Weights, ROOT::Math::Interpolation::kAKIMA);
          weight = my.Eval(theHF1EScale);
        }
      } else {
        threshold = theHF2threshold;
        weight = theHF2weight;
        if (weight <= 0.) {
          ROOT::Math::Interpolator my(theHF2Grid, theHF2Weights, ROOT::Math::Interpolation::kAKIMA);
          weight = my.Eval(theHF2EScale);
        }
      }
    }
  } else {
    edm::LogError("CaloTowersCreationAlgo") << "Bad cell: " << det << std::endl;
  }
}

void CaloTowersCreationAlgo::setEBEScale(double scale) {
  if (scale > 0.00001)
    *&theEBEScale = scale;
  else
    *&theEBEScale = 50.;
}

void CaloTowersCreationAlgo::setEEEScale(double scale) {
  if (scale > 0.00001)
    *&theEEEScale = scale;
  else
    *&theEEEScale = 50.;
}

void CaloTowersCreationAlgo::setHBEScale(double scale) {
  if (scale > 0.00001)
    *&theHBEScale = scale;
  else
    *&theHBEScale = 50.;
}

void CaloTowersCreationAlgo::setHESEScale(double scale) {
  if (scale > 0.00001)
    *&theHESEScale = scale;
  else
    *&theHESEScale = 50.;
}

void CaloTowersCreationAlgo::setHEDEScale(double scale) {
  if (scale > 0.00001)
    *&theHEDEScale = scale;
  else
    *&theHEDEScale = 50.;
}

void CaloTowersCreationAlgo::setHOEScale(double scale) {
  if (scale > 0.00001)
    *&theHOEScale = scale;
  else
    *&theHOEScale = 50.;
}

void CaloTowersCreationAlgo::setHF1EScale(double scale) {
  if (scale > 0.00001)
    *&theHF1EScale = scale;
  else
    *&theHF1EScale = 50.;
}

void CaloTowersCreationAlgo::setHF2EScale(double scale) {
  if (scale > 0.00001)
    *&theHF2EScale = scale;
  else
    *&theHF2EScale = 50.;
}

GlobalPoint CaloTowersCreationAlgo::emCrystalShwrPos(DetId detId, float fracDepth) {
  auto cellGeometry = theGeometry->getGeometry(detId);
  GlobalPoint point = cellGeometry->getPosition();  // face of the cell

  if (fracDepth <= 0)
    return point;
  if (fracDepth > 1)
    fracDepth = 1;

  const GlobalPoint& backPoint = cellGeometry->getBackPoint();
  point += fracDepth * (backPoint - point);

  return point;
}

GlobalPoint CaloTowersCreationAlgo::hadSegmentShwrPos(DetId detId, float fracDepth) {
  // same code as above
  return emCrystalShwrPos(detId, fracDepth);
}

GlobalPoint CaloTowersCreationAlgo::hadShwrPos(const std::vector<std::pair<DetId, float> >& metaContains,
                                               float fracDepth,
                                               double hadE) {
  // this is based on available RecHits, can lead to different actual depths if
  // hits in multi-depth towers are not all there
#ifdef EDM_ML_DEBUG
  std::cout << "hadShwrPos called with " << metaContains.size() << " elements and energy " << hadE << ":" << fracDepth
            << std::endl;
#endif
  if (hadE <= 0)
    return GlobalPoint(0, 0, 0);

  double hadX = 0.0;
  double hadY = 0.0;
  double hadZ = 0.0;

  int nConst = 0;

  std::vector<std::pair<DetId, float> >::const_iterator mc_it = metaContains.begin();
  for (; mc_it != metaContains.end(); ++mc_it) {
    if (mc_it->first.det() != DetId::Hcal)
      continue;
    // do not use HO for deirection calculations for now
    if (HcalDetId(mc_it->first).subdet() == HcalOuter)
      continue;
    ++nConst;

    GlobalPoint p = hadSegmentShwrPos(mc_it->first, fracDepth);

    // longitudinal segmentation: do not weight by energy,
    // get the geometrical position
    hadX += p.x();
    hadY += p.y();
    hadZ += p.z();
  }

  return GlobalPoint(hadX / nConst, hadY / nConst, hadZ / nConst);
}

GlobalPoint CaloTowersCreationAlgo::hadShwrPos(CaloTowerDetId towerId, float fracDepth) {
  // set depth using geometry of cells that are associated with the
  // tower (regardless if they have non-zero energies)

  //  if (hadE <= 0) return GlobalPoint(0, 0, 0);
#ifdef EDM_ML_DEBUG
  std::cout << "hadShwrPos " << towerId << " frac " << fracDepth << std::endl;
#endif
  if (fracDepth < 0)
    fracDepth = 0;
  else if (fracDepth > 1)
    fracDepth = 1;

  GlobalPoint point(0, 0, 0);

  int iEta = towerId.ieta();
  int iPhi = towerId.iphi();

  HcalDetId frontCellId, backCellId;

  if (towerId.ietaAbs() >= theTowerTopology->firstHFRing()) {
    // forward, take the geometry for long fibers
    frontCellId = HcalDetId(HcalForward, towerId.zside() * theTowerTopology->convertCTtoHcal(abs(iEta)), iPhi, 1);
    backCellId = HcalDetId(HcalForward, towerId.zside() * theTowerTopology->convertCTtoHcal(abs(iEta)), iPhi, 1);
  } else {
    //use constituents map
    std::vector<DetId> items = theTowerConstituentsMap->constituentsOf(towerId);
    int frontDepth = 1000;
    int backDepth = -1000;
    for (unsigned i = 0; i < items.size(); i++) {
      if (items[i].det() != DetId::Hcal)
        continue;
      HcalDetId hid(items[i]);
      if (hid.subdet() == HcalOuter)
        continue;
      if (!theHcalTopology->validHcal(hid, 2))
        continue;

      if (theHcalTopology->idFront(hid).depth() < frontDepth) {
        frontCellId = hid;
        frontDepth = theHcalTopology->idFront(hid).depth();
      }
      if (theHcalTopology->idBack(hid).depth() > backDepth) {
        backCellId = hid;
        backDepth = theHcalTopology->idBack(hid).depth();
      }
    }
#ifdef EDM_ML_DEBUG
    std::cout << "Front " << frontCellId << " Back " << backCellId << " Depths " << frontDepth << ":" << backDepth
              << std::endl;
#endif
    //fix for tower 28/29 - no tower 29 at highest depths
    if (towerId.ietaAbs() == theTowerTopology->lastHERing() && (theHcalPhase == 0 || theHcalPhase == 1)) {
      CaloTowerDetId towerId28(towerId.ieta() - towerId.zside(), towerId.iphi());
      std::vector<DetId> items28 = theTowerConstituentsMap->constituentsOf(towerId28);
#ifdef EDM_ML_DEBUG
      std::cout << towerId28 << " with " << items28.size() << " constituents:";
      for (unsigned k = 0; k < items28.size(); ++k)
        if (items28[k].det() == DetId::Hcal)
          std::cout << " " << HcalDetId(items28[k]);
      std::cout << std::endl;
#endif
      for (unsigned i = 0; i < items28.size(); i++) {
        if (items28[i].det() != DetId::Hcal)
          continue;
        HcalDetId hid(items28[i]);
        if (hid.subdet() == HcalOuter)
          continue;

        if (theHcalTopology->idBack(hid).depth() > backDepth) {
          backCellId = hid;
          backDepth = theHcalTopology->idBack(hid).depth();
        }
      }
    }
#ifdef EDM_ML_DEBUG
    std::cout << "Back " << backDepth << " ID " << backCellId << std::endl;
#endif
  }
  point = hadShwPosFromCells(DetId(frontCellId), DetId(backCellId), fracDepth);

  return point;
}

GlobalPoint CaloTowersCreationAlgo::hadShwPosFromCells(DetId frontCellId, DetId backCellId, float fracDepth) {
  // uses the "front" and "back" cells
  // to determine the axis. point set by the predefined depth.

  HcalDetId hid1(frontCellId), hid2(backCellId);
  if (theHcalTopology->getMergePositionFlag()) {
    hid1 = theHcalTopology->idFront(frontCellId);
#ifdef EDM_ML_DEBUG
    std::cout << "Front " << HcalDetId(frontCellId) << " " << hid1 << "\n";
#endif
    hid2 = theHcalTopology->idBack(backCellId);
#ifdef EDM_ML_DEBUG
    std::cout << "Back  " << HcalDetId(backCellId) << " " << hid2 << "\n";
#endif
  }

  auto frontCellGeometry = theGeometry->getGeometry(DetId(hid1));
  auto backCellGeometry = theGeometry->getGeometry(DetId(hid2));

  GlobalPoint point = frontCellGeometry->getPosition();
  const GlobalPoint& backPoint = backCellGeometry->getBackPoint();

  point += fracDepth * (backPoint - point);

  return point;
}

GlobalPoint CaloTowersCreationAlgo::emShwrPos(const std::vector<std::pair<DetId, float> >& metaContains,
                                              float fracDepth,
                                              double emE) {
  if (emE <= 0)
    return GlobalPoint(0, 0, 0);

  double emX = 0.0;
  double emY = 0.0;
  double emZ = 0.0;

  double eSum = 0;

  std::vector<std::pair<DetId, float> >::const_iterator mc_it = metaContains.begin();
  for (; mc_it != metaContains.end(); ++mc_it) {
    if (mc_it->first.det() != DetId::Ecal)
      continue;
    GlobalPoint p = emCrystalShwrPos(mc_it->first, fracDepth);
    double e = mc_it->second;

    if (e > 0) {
      emX += p.x() * e;
      emY += p.y() * e;
      emZ += p.z() * e;
      eSum += e;
    }
  }

  return GlobalPoint(emX / eSum, emY / eSum, emZ / eSum);
}

GlobalPoint CaloTowersCreationAlgo::emShwrLogWeightPos(const std::vector<std::pair<DetId, float> >& metaContains,
                                                       float fracDepth,
                                                       double emE) {
  double emX = 0.0;
  double emY = 0.0;
  double emZ = 0.0;

  double weight = 0;
  double sumWeights = 0;
  double sumEmE = 0;  // add crystals with E/E_EM > 1.5%
  double crystalThresh = 0.015 * emE;

  std::vector<std::pair<DetId, float> >::const_iterator mc_it = metaContains.begin();
  for (; mc_it != metaContains.end(); ++mc_it) {
    if (mc_it->second < 0)
      continue;
    if (mc_it->first.det() == DetId::Ecal && mc_it->second > crystalThresh)
      sumEmE += mc_it->second;
  }

  for (mc_it = metaContains.begin(); mc_it != metaContains.end(); ++mc_it) {
    if (mc_it->first.det() != DetId::Ecal || mc_it->second < crystalThresh)
      continue;

    GlobalPoint p = emCrystalShwrPos(mc_it->first, fracDepth);

    weight = 4.2 + log(mc_it->second / sumEmE);
    sumWeights += weight;

    emX += p.x() * weight;
    emY += p.y() * weight;
    emZ += p.z() * weight;
  }

  return GlobalPoint(emX / sumWeights, emY / sumWeights, emZ / sumWeights);
}

int CaloTowersCreationAlgo::compactTime(float time) {
  const float timeUnit = 0.01;  // discretization (ns)

  if (time > 300.0)
    return 30000;
  if (time < -300.0)
    return -30000;

  return int(time / timeUnit + 0.5);
}

//========================================================
//
// Bad/anomolous cell handling

void CaloTowersCreationAlgo::makeHcalDropChMap() {
  // This method fills the map of number of dead channels for the calotower,
  // The key of the map is CaloTowerDetId.
  // By definition these channels are not going to be in the RecHit collections.
  hcalDropChMap.clear();
  std::vector<DetId> allChanInStatusCont = theHcalChStatus->getAllChannels();

#ifdef EDM_ML_DEBUG
  std::cout << "DropChMap with " << allChanInStatusCont.size() << " channels" << std::endl;
#endif
  for (std::vector<DetId>::iterator it = allChanInStatusCont.begin(); it != allChanInStatusCont.end(); ++it) {
    const uint32_t dbStatusFlag = theHcalChStatus->getValues(*it)->getValue();
    if (theHcalSevLvlComputer->dropChannel(dbStatusFlag)) {
      DetId id = theHcalTopology->mergedDepthDetId(HcalDetId(*it));

      CaloTowerDetId twrId = theTowerConstituentsMap->towerOf(id);

      hcalDropChMap[twrId].first += 1;

      HcalDetId hid(*it);

      // special case for tower 29: if HCAL hit is in depth 3 add to twr 29 as well
      if (hid.subdet() == HcalEndcap && (theHcalPhase == 0 || theHcalPhase == 1) &&
          hid.ietaAbs() == theHcalTopology->lastHERing() - 1) {
        bool merge = theHcalTopology->mergedDepth29(hid);
        if (merge) {
          CaloTowerDetId twrId29(twrId.ieta() + twrId.zside(), twrId.iphi());
          hcalDropChMap[twrId29].first += 1;
        }
      }
    }
  }
  // now I know how many bad channels, but I also need to know if there's any good ones
  if (missingHcalRescaleFactorForEcal > 0) {
    for (auto& pair : hcalDropChMap) {
      if (pair.second.first == 0)
        continue;  // unexpected, but just in case
      int ngood = 0, nbad = 0;
      for (DetId id : theTowerConstituentsMap->constituentsOf(pair.first)) {
        if (id.det() != DetId::Hcal)
          continue;
        HcalDetId hid(id);
        if (hid.subdet() != HcalBarrel && hid.subdet() != HcalEndcap)
          continue;
        const uint32_t dbStatusFlag = theHcalChStatus->getValues(id)->getValue();
        if (dbStatusFlag == 0 || !theHcalSevLvlComputer->dropChannel(dbStatusFlag)) {
          ngood += 1;
        } else {
          nbad += 1;  // recount, since pair.second.first may include HO
        }
      }
      if (nbad > 0 && nbad >= ngood) {
        //uncomment for debug (may be useful to tune the criteria above)
        //CaloTowerDetId id(pair.first);
        //std::cout << "CaloTower at ieta = " << id.ieta() << ", iphi " << id.iphi() << ": set Hcal as not efficient (ngood =" << ngood << ", nbad = " << nbad << ")" << std::endl;
        pair.second.second = true;
      }
    }
  }
}

void CaloTowersCreationAlgo::makeEcalBadChs() {
  // std::cout << "VI making EcalBadChs ";

  // for ECAL the number of all bad channels is obtained here -----------------------

  for (auto ind = 0U; ind < theTowerTopology->sizeForDenseIndexing(); ++ind) {
    auto& numBadEcalChan = ecalBadChs[ind];
    numBadEcalChan = 0;
    auto id = theTowerTopology->detIdFromDenseIndex(ind);

    // this is utterly slow... (can be optmized if really needed)

    // get all possible constituents of the tower
    std::vector<DetId> allConstituents = theTowerConstituentsMap->constituentsOf(id);

    for (std::vector<DetId>::iterator ac_it = allConstituents.begin(); ac_it != allConstituents.end(); ++ac_it) {
      if (ac_it->det() != DetId::Ecal)
        continue;

      auto thisEcalSevLvl = theEcalSevLvlAlgo->severityLevel(*ac_it);

      // check if the Ecal severity is ok to keep
      std::vector<int>::const_iterator sevit =
          std::find(theEcalSeveritiesToBeExcluded.begin(), theEcalSeveritiesToBeExcluded.end(), thisEcalSevLvl);
      if (sevit != theEcalSeveritiesToBeExcluded.end()) {
        ++numBadEcalChan;
      }
    }

    // if (0!=numBadEcalChan) std::cout << id << ":" << numBadEcalChan << ", ";
  }

  /*
  int tot=0;
  for (auto ind=0U; ind<theTowerTopology->sizeForDenseIndexing(); ++ind) {
    if (ecalBadChs[ind]!=0) ++tot; 
  }
  std::cout << " | " << tot << std::endl;
  */
}

//////  Get status of the channel contributing to the tower

unsigned int CaloTowersCreationAlgo::hcalChanStatusForCaloTower(const CaloRecHit* hit) {
  HcalDetId hid(hit->detid());
  DetId id = theHcalTopology->idFront(hid);
#ifdef EDM_ML_DEBUG
  std::cout << "ChanStatusForCaloTower for " << hid << " to " << HcalDetId(id) << std::endl;
#endif
  const uint32_t recHitFlag = hit->flags();
  const uint32_t dbStatusFlag = theHcalChStatus->getValues(id)->getValue();

  int severityLevel = theHcalSevLvlComputer->getSeverityLevel(id, recHitFlag, dbStatusFlag);
  bool isRecovered = theHcalSevLvlComputer->recoveredRecHit(id, recHitFlag);

  // For use with hits rejected in the default reconstruction
  if (useRejectedHitsOnly) {
    if (!isRecovered) {
      if (severityLevel <= int(theHcalAcceptSeverityLevel) ||
          severityLevel > int(theHcalAcceptSeverityLevelForRejectedHit))
        return CaloTowersCreationAlgo::IgnoredChan;
      // this hit was either already accepted or is worse than
    } else {
      if (theRecoveredHcalHitsAreUsed || !useRejectedRecoveredHcalHits) {
        // skip recovered hits either because they were already used or because there was an explicit instruction
        return CaloTowersCreationAlgo::IgnoredChan;
      } else if (useRejectedRecoveredHcalHits) {
        return CaloTowersCreationAlgo::RecoveredChan;
      }

    }  // recovered channels

    // clasify channels as problematic: no good hits are supposed to be present in the
    // extra rechit collections
    return CaloTowersCreationAlgo::ProblematicChan;

  }  // treatment of rejected hits

  // this is for the regular reconstruction sequence

  if (severityLevel == 0)
    return CaloTowersCreationAlgo::GoodChan;

  if (isRecovered) {
    return (theRecoveredHcalHitsAreUsed) ? CaloTowersCreationAlgo::RecoveredChan : CaloTowersCreationAlgo::BadChan;
  } else {
    if (severityLevel > int(theHcalAcceptSeverityLevel)) {
      return CaloTowersCreationAlgo::BadChan;
    } else {
      return CaloTowersCreationAlgo::ProblematicChan;
    }
  }
}

std::tuple<unsigned int, bool> CaloTowersCreationAlgo::ecalChanStatusForCaloTower(const EcalRecHit* hit) {
  // const DetId id = hit->detid();

  //  uint16_t dbStatus = theEcalChStatus->find(id)->getStatusCode();
  //  uint32_t rhFlags  = hit->flags();
  //  int severityLevel = theEcalSevLvlAlgo->severityLevel(rhFlags, dbStatus);
  // The methods above will become private and cannot be usef for flagging ecal spikes.
  // Use the recommended interface - we leave the parameters for spilke removal to be specified by ECAL.

  //  int severityLevel = 999;

  EcalRecHit const& rh = *reinterpret_cast<EcalRecHit const*>(hit);
  int severityLevel = theEcalSevLvlAlgo->severityLevel(rh);

  //  if      (id.subdetId() == EcalBarrel) severityLevel = theEcalSevLvlAlgo->severityLevel( id, *theEbHandle);//, *theEcalChStatus);
  //  else if (id.subdetId() == EcalEndcap) severityLevel = theEcalSevLvlAlgo->severityLevel( id, *theEeHandle);//, *theEcalChStatus);

  // there should be no other ECAL types used in this reconstruction

  // The definition of ECAL severity levels uses categories that
  // are similar to the defined for CaloTower. (However, the categorization
  // for CaloTowers depends on the specified maximum acceptabel severity and therefore cannnot
  // be exact correspondence between the two. ECAL has additional categories describing modes of failure.)
  // This approach is different from the initial idea and from
  // the implementation for HCAL. Still make the logic similar to HCAL so that one has the ability to
  // exclude problematic channels as defined by ECAL.
  // For definitions of ECAL severity levels see RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h

  bool isBad = (severityLevel == EcalSeverityLevel::kBad);

  bool isRecovered = (severityLevel == EcalSeverityLevel::kRecovered);

  // check if the severity is compatible with our configuration
  // This applies to the "default" tower cleaning
  std::vector<int>::const_iterator sevit =
      std::find(theEcalSeveritiesToBeExcluded.begin(), theEcalSeveritiesToBeExcluded.end(), severityLevel);
  bool accepted = (sevit == theEcalSeveritiesToBeExcluded.end());

  // For use with hits that were rejected in the regular reconstruction:
  // This is for creating calotowers with lower level of cleaning by merging
  // the information from the default towers and a collection of towers created from
  // bad rechits

  if (useRejectedHitsOnly) {
    if (!isRecovered) {
      if (accepted || std::find(theEcalSeveritiesToBeUsedInBadTowers.begin(),
                                theEcalSeveritiesToBeUsedInBadTowers.end(),
                                severityLevel) == theEcalSeveritiesToBeUsedInBadTowers.end())
        return std::make_tuple(CaloTowersCreationAlgo::IgnoredChan, isBad);
      // this hit was either already accepted, or is not eligible for inclusion
    } else {
      if (theRecoveredEcalHitsAreUsed || !useRejectedRecoveredEcalHits) {
        // skip recovered hits either because they were already used or because there was an explicit instruction
        return std::make_tuple(CaloTowersCreationAlgo::IgnoredChan, isBad);
        ;
      } else if (useRejectedRecoveredEcalHits) {
        return std::make_tuple(CaloTowersCreationAlgo::RecoveredChan, isBad);
      }

    }  // recovered channels

    // clasify channels as problematic
    return std::make_tuple(CaloTowersCreationAlgo::ProblematicChan, isBad);

  }  // treatment of rejected hits

  // for normal reconstruction
  if (severityLevel == EcalSeverityLevel::kGood)
    return std::make_tuple(CaloTowersCreationAlgo::GoodChan, false);

  if (isRecovered) {
    return std::make_tuple(
        (theRecoveredEcalHitsAreUsed) ? CaloTowersCreationAlgo::RecoveredChan : CaloTowersCreationAlgo::BadChan, true);
  } else {
    return std::make_tuple(accepted ? CaloTowersCreationAlgo::ProblematicChan : CaloTowersCreationAlgo::BadChan, isBad);
  }
}
