#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GhostBusterPreferRefDt.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFConfiguration.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <sstream>

AlgoMuons GhostBusterPreferRefDt::select(AlgoMuons muonsIN, int charge) {
  // sorting within GB.
  //this function is only for the OMTF version without unconstrained pt
  auto customLess = [&](const AlgoMuons::value_type& a, const AlgoMuons::value_type& b) -> bool {
    if (!a->isValid()) {
      return true;
    }
    if (!b->isValid()) {
      return false;
    }

    int aRefLayerLogicNum = omtfConfig->getRefToLogicNumber()[a->getRefLayer()];
    int bRefLayerLogicNum = omtfConfig->getRefToLogicNumber()[b->getRefLayer()];
    if (a->getFiredLayerCntConstr() > b->getFiredLayerCntConstr())
      return false;
    else if (a->getFiredLayerCntConstr() == b->getFiredLayerCntConstr() && aRefLayerLogicNum < bRefLayerLogicNum) {
      return false;
    } else if (a->getFiredLayerCntConstr() == b->getFiredLayerCntConstr() && aRefLayerLogicNum == bRefLayerLogicNum &&
               a->getPdfSumConstr() > b->getPdfSumConstr())
      return false;
    else if (a->getFiredLayerCntConstr() == b->getFiredLayerCntConstr() && aRefLayerLogicNum == bRefLayerLogicNum &&
             a->getPdfSumConstr() == b->getPdfSumConstr() && a->getPatternNumConstr() > b->getPatternNumConstr())
      return false;
    else if (a->getFiredLayerCntConstr() == b->getFiredLayerCntConstr() && aRefLayerLogicNum == bRefLayerLogicNum &&
             a->getPdfSumConstr() == b->getPdfSumConstr() && a->getPatternNumConstr() == b->getPatternNumConstr() &&
             a->getRefHitNumber() < b->getRefHitNumber())
      return false;
    else
      return true;
  };

  auto customLessByFPLLH = [&](const AlgoMuons::value_type& a, const AlgoMuons::value_type& b) -> bool {
    if (!a->isValid()) {
      return true;
    }
    if (!b->isValid()) {
      return false;
    }

    if (a->getFiredLayerCntConstr() > b->getFiredLayerCntConstr())
      return false;
    else if (a->getFiredLayerCntConstr() == b->getFiredLayerCntConstr() && a->getPdfSumConstr() > b->getPdfSumConstr())
      return false;
    else if (a->getFiredLayerCntConstr() == b->getFiredLayerCntConstr() &&
             a->getPdfSumConstr() == b->getPdfSumConstr() && a->getPatternNumConstr() > b->getPatternNumConstr())
      return false;
    else if (a->getFiredLayerCntConstr() == b->getFiredLayerCntConstr() &&
             a->getPdfSumConstr() == b->getPdfSumConstr() && a->getPatternNumConstr() == b->getPatternNumConstr() &&
             a->getRefHitNumber() < b->getRefHitNumber())
      return false;
    else
      return true;
  };

  auto customLessByLLH = [&](const AlgoMuons::value_type& a, const AlgoMuons::value_type& b) -> bool {
    if (!a->isValid()) {
      return true;
    }
    if (!b->isValid()) {
      return false;
    }

    if (a->getPdfSumConstr() > b->getPdfSumConstr())
      return false;
    else if (a->getPdfSumConstr() == b->getPdfSumConstr() && a->getPatternNumConstr() > b->getPatternNumConstr())
      return false;
    else if (a->getPdfSumConstr() == b->getPdfSumConstr() && a->getPatternNumConstr() == b->getPatternNumConstr() &&
             a->getRefHitNumber() < b->getRefHitNumber())
      return false;
    else
      return true;
  };

  //this function is for the OMTF version with unconstrained pt
  auto customByRefLayer = [&](const AlgoMuons::value_type& a, const AlgoMuons::value_type& b) -> bool {
    if (!a->isValid()) {
      return true;
    }
    if (!b->isValid()) {
      return false;
    }

    int aRefLayerLogicNum = omtfConfig->getRefToLogicNumber()[a->getRefLayer()];
    int bRefLayerLogicNum = omtfConfig->getRefToLogicNumber()[b->getRefLayer()];

    if (aRefLayerLogicNum < bRefLayerLogicNum) {
      return false;
    } else if (aRefLayerLogicNum == bRefLayerLogicNum && a->getPdfSum() > b->getPdfSum())
      return false;
    else if (aRefLayerLogicNum == bRefLayerLogicNum && a->getPdfSum() == b->getPdfSum() &&
             a->getPatternNum() > b->getPatternNum())
      return false;
    else if (aRefLayerLogicNum == bRefLayerLogicNum && a->getPdfSum() == b->getPdfSum() &&
             a->getPatternNum() == b->getPatternNum() && a->getRefHitNumber() < b->getRefHitNumber())
      return false;
    else
      return true;
  };

  if (omtfConfig->getGhostBusterType() == "byLLH")
    std::sort(muonsIN.rbegin(), muonsIN.rend(), customLessByLLH);
  else if (omtfConfig->getGhostBusterType() == "byFPLLH")
    std::sort(muonsIN.rbegin(), muonsIN.rend(), customLessByFPLLH);
  else if (omtfConfig->getGhostBusterType() == "byRefLayer")
    std::sort(muonsIN.rbegin(), muonsIN.rend(), customByRefLayer);
  else
    std::sort(muonsIN.rbegin(), muonsIN.rend(), customLess);

  // actual GhostBusting. Overwrite eta in case of no DT info.
  AlgoMuons refHitCleanCandsFixedEta;

  for (unsigned int iMu1 = 0; iMu1 < muonsIN.size(); iMu1++) {
    refHitCleanCandsFixedEta.emplace_back(new AlgoMuon(*(muonsIN[iMu1])));

    if (omtfConfig->getStubEtaEncoding() == ProcConfigurationBase::StubEtaEncoding::bits)
      refHitCleanCandsFixedEta.back()->setEta(OMTFConfiguration::etaBits2HwEta(muonsIN[iMu1]->getEtaHw()));
  }

  for (unsigned int iMu1 = 0; iMu1 < refHitCleanCandsFixedEta.size(); iMu1++) {
    auto& muIN1 = refHitCleanCandsFixedEta[iMu1];
    //watch out: the muIN1 is AlgoMuonPtr, so setting the eta here changes the eta in the input muonsIN
    //this affects algoCandidates in OMTFProcessor<GoldenPatternType>::run

    if (!muIN1->isValid() || muIN1->isKilled())
      continue;

    for (unsigned int iMu2 = refHitCleanCandsFixedEta.size() - 1; iMu2 >= iMu1 + 1; iMu2--) {
      auto& muIN2 = refHitCleanCandsFixedEta[iMu2];
      if (muIN2->isValid() &&
          std::abs(omtfConfig->procPhiToGmtPhi(muIN1->getPhi()) - omtfConfig->procPhiToGmtPhi(muIN2->getPhi())) < 8) {
        //the candidates are sorted, so only the  muIN2 can be killed, as it is "worse" than the muIN1
        refHitCleanCandsFixedEta[iMu2]->kill();
        refHitCleanCandsFixedEta[iMu1]->getKilledMuons().emplace_back(muIN2);

        //for the DT stubs, if there is no eta, the middle of the chamber is set as the stub eta, i.e. 75, 79 or 92 respectively
        //in this case the eta can be replaced by the eta from the killed algoMuon
        if (omtfConfig->getRefToLogicNumber()[muIN1->getRefLayer()] <= 5 && (omtfConfig->fwVersion() >= 6) &&
            ((abs(muIN1->getEtaHw()) == 75 || abs(muIN1->getEtaHw()) == 79 || abs(muIN1->getEtaHw()) == 92)) &&
            ((abs(muIN2->getEtaHw()) != 75 && abs(muIN2->getEtaHw()) != 79 &&
              abs(muIN2->getEtaHw()) != 92))) {  //FIXME condition in this do not affects the final result

          muIN1->setEta(muIN2->getEtaHw());
        }
      }
    }
  }

  // fill outgoing collection
  /* there is nowhere a cut on the pdfSum > 0 for a muon to be valid
   * muon is valid if getPtConstr() > 0 || getPtUnconstr() > 0,
   * i.e. when there was a fitting pattern
   * this mean there can be a muon with pdfSumConstrained = 0 but with not 0 PtConstr
   * which is OK. See also comment in the GoldenPatternResult::finalise10()
   */
  AlgoMuons refHitCleanCands;
  for (const auto& mu : refHitCleanCandsFixedEta) {
    if (mu->isValid() && !(mu->isKilled()))
      refHitCleanCands.emplace_back(mu);
    if (refHitCleanCands.size() >= 3)
      break;
  }

  while (refHitCleanCands.size() < 3)
    refHitCleanCands.emplace_back(new AlgoMuon());

  return refHitCleanCands;
}
