#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GhostBusterPreferRefDt.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFConfiguration.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <sstream>

namespace {

  struct AlgoMuonEtaFix : public AlgoMuon {
    AlgoMuonEtaFix(const AlgoMuon& mu) : AlgoMuon(mu), fixedEta(mu.getEtaHw()) {}
    unsigned int fixedEta;
  };

}  // namespace

AlgoMuons GhostBusterPreferRefDt::select(AlgoMuons muonsIN, int charge) {
  // sorting within GB.
  auto customLess = [&](const AlgoMuons::value_type& a, const AlgoMuons::value_type& b) -> bool {
    if (!a->isValid()) {
      return true;
    }
    if (!b->isValid()) {
      return false;
    }

    int aRefLayerLogicNum = omtfConfig->getRefToLogicNumber()[a->getRefLayer()];
    int bRefLayerLogicNum = omtfConfig->getRefToLogicNumber()[b->getRefLayer()];
    if (a->getQ() > b->getQ())
      return false;
    else if (a->getQ() == b->getQ() && aRefLayerLogicNum < bRefLayerLogicNum) {
      return false;
    } else if (a->getQ() == b->getQ() && aRefLayerLogicNum == bRefLayerLogicNum && a->getDisc() > b->getDisc())
      return false;
    else if (a->getQ() == b->getQ() && aRefLayerLogicNum == bRefLayerLogicNum && a->getDisc() == b->getDisc() &&
             a->getPatternNumber() > b->getPatternNumber())
      return false;
    else if (a->getQ() == b->getQ() && aRefLayerLogicNum == bRefLayerLogicNum && a->getDisc() == b->getDisc() &&
             a->getPatternNumber() == b->getPatternNumber() && a->getRefHitNumber() < b->getRefHitNumber())
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

    if (a->getQ() > b->getQ())
      return false;
    else if (a->getQ() == b->getQ()) {
      return false;
    } else if (a->getQ() == b->getQ() && a->getDisc() > b->getDisc())
      return false;
    else if (a->getQ() == b->getQ() && a->getDisc() == b->getDisc() && a->getPatternNumber() > b->getPatternNumber())
      return false;
    else if (a->getQ() == b->getQ() && a->getDisc() == b->getDisc() && a->getPatternNumber() == b->getPatternNumber() &&
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

    if (a->getDisc() > b->getDisc())
      return false;
    else if (a->getDisc() == b->getDisc() && a->getPatternNumber() > b->getPatternNumber())
      return false;
    else if (a->getDisc() == b->getDisc() && a->getPatternNumber() == b->getPatternNumber() &&
             a->getRefHitNumber() < b->getRefHitNumber())
      return false;
    else
      return true;
  };

  /*
  auto customLess = [&](const AlgoMuons::value_type& a, const AlgoMuons::value_type& b)->bool {
    if(!a->isValid()) {
      return true;
    }
    if(!b->isValid()) {
      return false;
    }

    int aRefLayerLogicNum = omtfConfig->getRefToLogicNumber()[a->getRefLayer()];
    int bRefLayerLogicNum = omtfConfig->getRefToLogicNumber()[b->getRefLayer()];
    // if(a->getQ() > b->getQ())
    //   return false;
    if(aRefLayerLogicNum < bRefLayerLogicNum) {
      return false;
    }
    else if (aRefLayerLogicNum == bRefLayerLogicNum && a->getDisc() > b->getDisc() )
      return false;
    else if (aRefLayerLogicNum == bRefLayerLogicNum && a->getDisc() == b->getDisc() && a->getPatternNumber() > b->getPatternNumber() )
      return false;
    else if (aRefLayerLogicNum == bRefLayerLogicNum && a->getDisc() == b->getDisc() && a->getPatternNumber() == b->getPatternNumber() && a->getRefHitNumber() < b->getRefHitNumber())
      return false;
    else
      return true;
  };*/

  if (omtfConfig->getGhostBusterType() == "byLLH")
    std::sort(muonsIN.rbegin(), muonsIN.rend(), customLessByLLH);
  else if (omtfConfig->getGhostBusterType() == "byFPLLH")
    std::sort(muonsIN.rbegin(), muonsIN.rend(), customLessByFPLLH);
  else
    std::sort(muonsIN.rbegin(), muonsIN.rend(), customLess);

  // actual GhostBusting. Overwrite eta in case of no DT info.
  std::vector<AlgoMuonEtaFix> refHitCleanCandsFixedEta;
  for (const auto& muIN : muonsIN) {
    if (!muIN->isValid())
      continue;

    //edm::LogVerbatim("OMTFReconstruction") << "GhostBusting "<<*muIN<<" phiGMT "<<omtfConfig->procPhiToGmtPhi(muIN->getPhi())<< std::endl;

    refHitCleanCandsFixedEta.push_back(*muIN);  //FIXME to much copying here...
    auto killIt = refHitCleanCandsFixedEta.end();

    //do not accept candidates with similar phi (any charge combination)
    //veto window 5 degree in GMT scale is 5/360*576=8 units
    for (auto it1 = refHitCleanCandsFixedEta.begin(); it1 != refHitCleanCandsFixedEta.end(); ++it1) {
      for (auto it2 = std::next(it1); it2 != refHitCleanCandsFixedEta.end(); ++it2) {
        if (it2->isValid() &&
            std::abs(omtfConfig->procPhiToGmtPhi(it1->getPhi()) - omtfConfig->procPhiToGmtPhi(it2->getPhi())) < 8) {
          killIt = it2;
          if ((omtfConfig->fwVersion() >= 6) &&
              ((abs(it1->getEtaHw()) == 75 || abs(it1->getEtaHw()) == 79 || abs(it1->getEtaHw()) == 92)) &&
              ((abs(it2->getEtaHw()) != 75 && abs(it2->getEtaHw()) != 79 && abs(it2->getEtaHw()) != 92)))
            it1->fixedEta = it2->getEtaHw();
        }
      }
    }
    if (killIt != refHitCleanCandsFixedEta.end())
      refHitCleanCandsFixedEta.erase(killIt);
  }

  // fill outgoing collection
  AlgoMuons refHitCleanCands;
  for (const auto& mu : refHitCleanCandsFixedEta) {
    AlgoMuon fixed = mu;
    fixed.setEta(mu.fixedEta);
    refHitCleanCands.emplace_back(new AlgoMuon(fixed));
  }

  while (refHitCleanCands.size() < 3)
    refHitCleanCands.emplace_back(new AlgoMuon());

  /*
  std::stringstream myStr;
  bool hasCandidates = false;
  for(unsigned int iRefHit=0;iRefHit<refHitCleanCands.size();++iRefHit){
    if(refHitCleanCands[iRefHit].getQ()){
      hasCandidates=true;
      break;
    }
  }
  for(unsigned int iRefHit=0;iRefHit<refHitCleanCands.size();++iRefHit){
    if(refHitCleanCands[iRefHit].getQ()) myStr<<"Ref hit: "<<iRefHit<<" "<<refHitCleanCands[iRefHit]<<std::endl;
  }
  if(hasCandidates) edm::LogInfo("OMTF Sorter")<<myStr.str();
*/

  return refHitCleanCands;
}
