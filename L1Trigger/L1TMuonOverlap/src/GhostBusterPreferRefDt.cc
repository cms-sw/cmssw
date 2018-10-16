#include "L1Trigger/L1TMuonOverlap/interface/GhostBusterPreferRefDt.h"

#include <sstream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFConfiguration.h"

namespace { 

  int phiGMT(int phiAlgo) { return phiAlgo*437/pow(2,12); }

  struct AlgoMuonEtaFix : public AlgoMuon {
    AlgoMuonEtaFix(const AlgoMuon & mu) : AlgoMuon(mu), fixedEta(mu.getEta()) {}
    unsigned int fixedEta;
  };

}

std::vector<AlgoMuon> GhostBusterPreferRefDt::select(std::vector<AlgoMuon> muonsIN, int charge) {

  // sorting within GB.
  auto customLess = [&](const AlgoMuon& a, const AlgoMuon& b)->bool {
    int aRefLayerLogicNum = omtfConfig->getRefToLogicNumber()[a.getRefLayer()];
    int bRefLayerLogicNum = omtfConfig->getRefToLogicNumber()[b.getRefLayer()];
    if(a.getQ() > b.getQ())
      return false;
    else if(a.getQ()==b.getQ() && aRefLayerLogicNum < bRefLayerLogicNum) {
      return false;
    }
    else if (a.getQ()==b.getQ() && aRefLayerLogicNum == bRefLayerLogicNum && a.getDisc() > b.getDisc() )
      return false;
    else if (a.getQ()==b.getQ() && aRefLayerLogicNum == bRefLayerLogicNum && a.getDisc() == b.getDisc() && a.getPatternNumber() > b.getPatternNumber() )
      return false;
    else if (a.getQ()==b.getQ() && aRefLayerLogicNum == bRefLayerLogicNum && a.getDisc() == b.getDisc() && a.getPatternNumber() == b.getPatternNumber() && a.getRefHitNumber() < b.getRefHitNumber())
      return false;
    else
      return true;
  };

  std::sort( muonsIN.rbegin(), muonsIN.rend(), customLess);

  // actual GhostBusting. Overwrite eta in case of no DT info.
  std::vector<AlgoMuonEtaFix> refHitCleanCandsFixedEta;
  for (const auto & muIN : muonsIN) {
    refHitCleanCandsFixedEta.push_back(muIN);
    auto killIt = refHitCleanCandsFixedEta.end();

    //do not accept candidates with similar phi (any charge combination)
    //veto window 5 degree in GMT scale is 5/360*576=8 units
    for (auto it1 = refHitCleanCandsFixedEta.begin(); it1 != refHitCleanCandsFixedEta.end(); ++it1) {
      for (auto it2 = std::next(it1); it2 != refHitCleanCandsFixedEta.end(); ++it2) {
        if (std::abs( phiGMT(it1->getPhi()) - phiGMT(it2->getPhi()) ) < 8 ) { 
          killIt = it2;
          if (    (omtfConfig->fwVersion() >= 6)
               && ((abs(it1->getEta())==75 || abs(it1->getEta())==79 || abs(it1->getEta())==92))
               && ((abs(it2->getEta())!=75 && abs(it2->getEta())!=79 && abs(it2->getEta())!=92)) ) it1->fixedEta=it2->getEta();
        }
      }
    } 
    if (killIt != refHitCleanCandsFixedEta.end()) refHitCleanCandsFixedEta.erase(killIt);
  }

  // fill outgoing collection 
  std::vector<AlgoMuon> refHitCleanCands;
  for (const auto & mu : refHitCleanCandsFixedEta) {
    AlgoMuon fixed = mu;
    fixed.setEta(mu.fixedEta);
    refHitCleanCands.push_back(fixed);
  } 

  refHitCleanCands.resize( 3, AlgoMuon(0,999,9999,0,0,0,0,0) ); 
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
