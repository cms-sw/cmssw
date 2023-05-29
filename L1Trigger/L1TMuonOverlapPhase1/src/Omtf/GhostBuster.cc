#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GhostBuster.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFConfiguration.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <sstream>

AlgoMuons GhostBuster::select(AlgoMuons refHitCands, int charge) {
  //edm::LogImportant("OMTFReconstruction")<<"calling "<<__PRETTY_FUNCTION__ <<std::endl;

  AlgoMuons refHitCleanCands;
  // Sort candidates with decreased goodness,
  auto customLess = [&](const AlgoMuons::value_type& a, const AlgoMuons::value_type& b) -> bool {
    return (*a) < (*b);  //< operator of AlgoMuon
  };

  std::sort(refHitCands.rbegin(), refHitCands.rend(), customLess);

  for (AlgoMuons::iterator it1 = refHitCands.begin(); it1 != refHitCands.end(); ++it1) {
    bool isGhost = false;
    for (AlgoMuons::iterator it2 = refHitCleanCands.begin(); it2 != refHitCleanCands.end(); ++it2) {
      //do not accept candidates with similar phi (any charge combination)
      //veto window 5deg(=half of logic cone)=5/360*5760=80"logic strips"
      //veto window 5 degree in GMT scale is 5/360*576=8 units
      if (std::abs(omtfConfig->procPhiToGmtPhi((*it1)->getPhi()) - omtfConfig->procPhiToGmtPhi((*it2)->getPhi())) < 8) {
        //      if(std::abs(it1->getPhi() - it2->getPhi())<5/360.0*nPhiBins){
        isGhost = true;
        break;
        //which one candidate is killed depends only on the order in the refHitCands (the one with smaller index is taken), and this order is assured by the sort above
        //TODO here the candidate that is killed does not kill other candidates - check if the firmware does the same (KB)
      }
    }
    if ((*it1)->getQ() > 0 && !isGhost)
      refHitCleanCands.emplace_back(new AlgoMuon(**it1));

    if (refHitCleanCands.size() >= 3)
      break;
  }

  while (refHitCleanCands.size() < 3)
    refHitCleanCands.emplace_back(new AlgoMuon());

  std::stringstream myStr;
  bool hasCandidates = false;
  for (unsigned int iRefHit = 0; iRefHit < refHitCands.size(); ++iRefHit) {
    if (refHitCands[iRefHit]->getQ()) {
      hasCandidates = true;
      break;
    }
  }
  for (unsigned int iRefHit = 0; iRefHit < refHitCands.size(); ++iRefHit) {
    if (refHitCands[iRefHit]->getQ())
      myStr << "Ref hit: " << iRefHit << " " << refHitCands[iRefHit] << std::endl;
  }
  myStr << "Selected Candidates with charge: " << charge << std::endl;
  for (unsigned int iCand = 0; iCand < refHitCleanCands.size(); ++iCand) {
    myStr << "Cand: " << iCand << " " << refHitCleanCands[iCand] << std::endl;
  }

  if (hasCandidates)
    edm::LogInfo("OMTF Sorter") << myStr.str();

  // update refHitCands with refHitCleanCands
  return refHitCleanCands;
}
